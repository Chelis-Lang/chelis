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
    bincode::deserialize(bytes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_roundtrips() {
        let shell = ShellPackage {
            package: PackageId {
                name: "chelis-std".to_string(),
                version: "0.1.0".to_string(),
            },
            compiler: "=0.1.7".to_string(),
            modules: vec![ShellModule {
                module: "Std.Nn.Linear".to_string(),
                exports: vec![ShellSymbol {
                    name: "forward".to_string(),
                    kind: SymbolKind::Value,
                    type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()),
                    effects: Vec::new(),
                    has_body: true,
                }],
            }],
            dependencies: Vec::new(),
            archive_sha256: "deadbeef".to_string(),
        };

        let bytes = encode_shell(&shell).expect("encode shell");
        let decoded = decode_shell(&bytes).expect("decode shell");
        assert_eq!(decoded, shell);
    }
}
