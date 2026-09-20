//! Generated C declaration metadata and source/header agreement checks.

use std::collections::BTreeMap;
use std::fmt;

const SOURCE_NAME_PREFIX: &str = "/* chelis-source-name: ";
const SOURCE_NAME_SUFFIX: &str = " */";

/// One public function declaration from a generated C header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedDeclaration {
    source_name: String,
    symbol: String,
    declaration: String,
}

impl GeneratedDeclaration {
    /// The exact Chelis identity carried by the generated header metadata.
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// The exact C symbol declared by the generated header.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// The canonical generated C declaration, including its semicolon.
    pub fn declaration(&self) -> &str {
        &self.declaration
    }
}

/// Parsed generated declarations keyed by exact Chelis identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedHeader {
    declarations: BTreeMap<String, GeneratedDeclaration>,
}

impl GeneratedHeader {
    /// Parse a generated C header and require source-name metadata for every declaration.
    pub fn parse(header: &str) -> Result<Self, GeneratedHeaderError> {
        let mut pending_source_name: Option<String> = None;
        let mut declarations = BTreeMap::new();

        for (index, raw_line) in header.lines().enumerate() {
            let line_number = index + 1;
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(source_name) = line
                .strip_prefix(SOURCE_NAME_PREFIX)
                .and_then(|line| line.strip_suffix(SOURCE_NAME_SUFFIX))
            {
                if source_name.is_empty() {
                    return Err(GeneratedHeaderError::new(format!(
                        "line {line_number}: generated source name is empty"
                    )));
                }
                if pending_source_name
                    .replace(source_name.to_string())
                    .is_some()
                {
                    return Err(GeneratedHeaderError::new(format!(
                        "line {line_number}: generated source-name metadata has no declaration"
                    )));
                }
                continue;
            }
            if !line.ends_with(';') {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: expected a generated C declaration"
                )));
            }
            let source_name = pending_source_name.take().ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "line {line_number}: generated declaration is missing source-name metadata"
                ))
            })?;
            let symbol = declaration_symbol(line).ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "line {line_number}: cannot identify the declared C function"
                ))
            })?;
            let declaration = GeneratedDeclaration {
                source_name: source_name.clone(),
                symbol: symbol.to_string(),
                declaration: line.to_string(),
            };
            if declarations
                .insert(source_name.clone(), declaration)
                .is_some()
            {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: duplicate generated declaration metadata for `{source_name}`"
                )));
            }
        }

        if let Some(source_name) = pending_source_name {
            return Err(GeneratedHeaderError::new(format!(
                "generated source-name metadata for `{source_name}` has no declaration"
            )));
        }
        if declarations.is_empty() {
            return Err(GeneratedHeaderError::new(
                "generated header contains no declarations",
            ));
        }
        Ok(Self { declarations })
    }

    /// Return the declaration associated with one exact Chelis identity.
    pub fn declaration(&self, source_name: &str) -> Option<&GeneratedDeclaration> {
        self.declarations.get(source_name)
    }

    /// Require every generated declaration to have an exact matching source definition.
    pub fn validate_source(&self, source: &str) -> Result<(), GeneratedHeaderError> {
        for declaration in self.declarations.values() {
            let signature = declaration
                .declaration
                .strip_suffix(';')
                .expect("parsed declarations end in semicolons");
            let definition = format!("{signature} {{");
            if !source.lines().any(|line| line.trim() == definition) {
                return Err(GeneratedHeaderError::new(format!(
                    "generated source has no exact definition for `{}`: `{}`",
                    declaration.source_name, declaration.declaration
                )));
            }
        }
        Ok(())
    }
}

/// A malformed or stale generated declaration contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedHeaderError {
    message: String,
}

impl GeneratedHeaderError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GeneratedHeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for GeneratedHeaderError {}

pub(crate) fn render_declaration(source_name: &str, declaration: &str) -> String {
    format!("{SOURCE_NAME_PREFIX}{source_name}{SOURCE_NAME_SUFFIX}\n{declaration}")
}

fn declaration_symbol(declaration: &str) -> Option<&str> {
    let head = declaration.split_once('(')?.0.trim_end();
    head.rsplit_once(char::is_whitespace)
        .map(|(_, symbol)| symbol.trim_start_matches('*'))
        .filter(|symbol| !symbol.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{GeneratedHeader, render_declaration};

    #[test]
    fn generated_header_requires_metadata_and_exact_source_agreement() {
        let header = render_declaration("entry", "float chelis_fn_656e747279(float x);");
        let parsed = GeneratedHeader::parse(&header).expect("generated header");
        let declaration = parsed.declaration("entry").expect("entry metadata");
        assert_eq!(declaration.symbol(), "chelis_fn_656e747279");
        parsed
            .validate_source("float chelis_fn_656e747279(float x) {\n    return x;\n}\n")
            .expect("matching definition");

        assert!(GeneratedHeader::parse("float chelis_fn_656e747279(float x);").is_err());
        assert!(
            parsed
                .validate_source("float chelis_fn_656e747279(double x) {\n    return x;\n}\n")
                .is_err()
        );
    }
}
