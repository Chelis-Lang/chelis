//! Generated C declaration metadata and source/header agreement checks.

use std::collections::BTreeMap;
use std::fmt;

const SOURCE_NAME_PREFIX: &str = "/* chelis-source-name: ";
const SOURCE_NAME_SUFFIX: &str = " */";
const AUTHORED_EXPORT_PREFIX: &str = "/* chelis-authored-export: ";
const DIRECT_EXPORT_PREFIX: &str = "/* chelis-direct-export: ";

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
        let definitions = source_definitions(source)?;
        for (source_name, declaration) in &self.declarations {
            let definition = definitions.get(source_name).ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "generated source has no exported definition for `{source_name}`"
                ))
            })?;
            if declaration.symbol != definition.symbol
                || declaration.declaration != definition.declaration
            {
                return Err(GeneratedHeaderError::new(format!(
                    "generated header/source association disagrees for `{source_name}`: header `{}`, source `{}`",
                    declaration.declaration, definition.declaration
                )));
            }
            validate_export_symbol(definition)?;
        }
        if let Some(source_name) = definitions
            .keys()
            .find(|source_name| !self.declarations.contains_key(*source_name))
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated source exports `{source_name}` without a matching header declaration"
            )));
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

pub(crate) fn render_authored_export_marker(source_name: &str) -> String {
    format!("{AUTHORED_EXPORT_PREFIX}{source_name}{SOURCE_NAME_SUFFIX}")
}

pub(crate) fn render_direct_export_marker(source_name: &str) -> String {
    format!("{DIRECT_EXPORT_PREFIX}{source_name}{SOURCE_NAME_SUFFIX}")
}

fn declaration_symbol(declaration: &str) -> Option<&str> {
    let head = declaration.split_once('(')?.0.trim_end();
    head.rsplit_once(char::is_whitespace)
        .map(|(_, symbol)| symbol.trim_start_matches('*'))
        .filter(|symbol| !symbol.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportKind {
    Authored,
    Direct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceDefinition {
    kind: ExportKind,
    source_name: String,
    symbol: String,
    declaration: String,
}

fn source_definitions(
    source: &str,
) -> Result<BTreeMap<String, SourceDefinition>, GeneratedHeaderError> {
    let mut definitions = BTreeMap::new();
    let mut pending_export: Option<(ExportKind, String)> = None;
    let mut depth = 0_i64;
    let mut scan = CScanState::default();

    for (index, raw_line) in source.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if depth == 0 {
            if let Some(source_name) = export_marker(line, AUTHORED_EXPORT_PREFIX) {
                set_pending_export(
                    &mut pending_export,
                    ExportKind::Authored,
                    source_name,
                    line_number,
                )?;
            } else if let Some(source_name) = export_marker(line, DIRECT_EXPORT_PREFIX) {
                set_pending_export(
                    &mut pending_export,
                    ExportKind::Direct,
                    source_name,
                    line_number,
                )?;
            } else if let Some((declaration, symbol, is_static)) = definition_line(line) {
                if is_static {
                    if let Some((_, source_name)) = pending_export.take() {
                        return Err(GeneratedHeaderError::new(format!(
                            "line {line_number}: export metadata for `{source_name}` precedes a static definition"
                        )));
                    }
                } else if symbol == "main" {
                    if let Some((_, source_name)) = pending_export.take() {
                        return Err(GeneratedHeaderError::new(format!(
                            "line {line_number}: export metadata for `{source_name}` cannot describe the process entry `main`"
                        )));
                    }
                } else {
                    let (kind, source_name) = pending_export.take().ok_or_else(|| {
                        GeneratedHeaderError::new(format!(
                            "line {line_number}: external definition `{symbol}` is missing generated export metadata"
                        ))
                    })?;
                    let definition = SourceDefinition {
                        kind,
                        source_name: source_name.clone(),
                        symbol: symbol.to_string(),
                        declaration,
                    };
                    if definitions
                        .insert(source_name.clone(), definition)
                        .is_some()
                    {
                        return Err(GeneratedHeaderError::new(format!(
                            "line {line_number}: duplicate exported source definition for `{source_name}`"
                        )));
                    }
                }
            } else if pending_export.is_some() && !line.is_empty() {
                let (_, source_name) = pending_export.take().expect("checked pending export");
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: export metadata for `{source_name}` has no adjacent definition"
                )));
            }
        }
        depth += scan.brace_delta(raw_line);
        if depth < 0 {
            return Err(GeneratedHeaderError::new(format!(
                "line {line_number}: generated source closes an unmatched brace"
            )));
        }
    }

    if let Some((_, source_name)) = pending_export {
        return Err(GeneratedHeaderError::new(format!(
            "generated export metadata for `{source_name}` has no definition"
        )));
    }
    if depth != 0 {
        return Err(GeneratedHeaderError::new(
            "generated source has unbalanced braces",
        ));
    }
    Ok(definitions)
}

fn export_marker<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix)
        .and_then(|line| line.strip_suffix(SOURCE_NAME_SUFFIX))
        .filter(|source_name| !source_name.is_empty())
}

fn set_pending_export(
    pending_export: &mut Option<(ExportKind, String)>,
    kind: ExportKind,
    source_name: &str,
    line_number: usize,
) -> Result<(), GeneratedHeaderError> {
    if let Some((_, pending_name)) = pending_export.replace((kind, source_name.to_string())) {
        return Err(GeneratedHeaderError::new(format!(
            "line {line_number}: export metadata for `{pending_name}` has no definition"
        )));
    }
    Ok(())
}

fn definition_line(line: &str) -> Option<(String, String, bool)> {
    let signature = line.strip_suffix('{')?.trim_end();
    if signature.ends_with('=') {
        return None;
    }
    let declaration = format!("{signature};");
    let symbol = declaration_symbol(&declaration)?.to_string();
    let is_static = signature
        .split_ascii_whitespace()
        .any(|word| word == "static");
    Some((declaration, symbol, is_static))
}

fn validate_export_symbol(definition: &SourceDefinition) -> Result<(), GeneratedHeaderError> {
    let valid = match definition.kind {
        ExportKind::Direct => definition.symbol == definition.source_name,
        ExportKind::Authored if definition.source_name == "main" => {
            definition.symbol != "main" && definition.symbol.ends_with("__main")
        }
        ExportKind::Authored
            if chelis_types::is_linker_format_name(&definition.source_name)
                && chelis_types::demangle_ident(&definition.source_name) == "main" =>
        {
            definition.symbol == definition.source_name
        }
        ExportKind::Authored => {
            definition.symbol == encoded_authored_symbol(&definition.source_name)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(GeneratedHeaderError::new(format!(
            "generated source export `{}` uses non-canonical C symbol `{}`",
            definition.source_name, definition.symbol
        )))
    }
}

fn encoded_authored_symbol(source_name: &str) -> String {
    use std::fmt::Write as _;

    let mut symbol = String::from("chelis_fn_");
    for byte in source_name.bytes() {
        write!(symbol, "{byte:02x}").expect("writing to a String cannot fail");
    }
    symbol
}

#[derive(Debug, Default)]
struct CScanState {
    block_comment: bool,
    quote: Option<char>,
    escaped: bool,
}

impl CScanState {
    fn brace_delta(&mut self, line: &str) -> i64 {
        if !self.block_comment && self.quote.is_none() && line.trim_start().starts_with('#') {
            return 0;
        }
        let mut chars = line.chars().peekable();
        let mut delta = 0;
        while let Some(ch) = chars.next() {
            if self.block_comment {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    self.block_comment = false;
                }
                continue;
            }
            if let Some(quote) = self.quote {
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == quote {
                    self.quote = None;
                }
                continue;
            }
            if ch == '/' && chars.peek() == Some(&'*') {
                chars.next();
                self.block_comment = true;
            } else if ch == '/' && chars.peek() == Some(&'/') {
                break;
            } else if ch == '"' || ch == '\'' {
                self.quote = Some(ch);
            } else if ch == '{' {
                delta += 1;
            } else if ch == '}' {
                delta -= 1;
            }
        }
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::{GeneratedHeader, render_declaration};

    fn authored_definition(source_name: &str, declaration: &str, body: &str) -> String {
        format!(
            "/* chelis-authored-export: {source_name} */\n{} {{\n    {body}\n}}\n",
            declaration
                .strip_suffix(';')
                .expect("test declaration ends in a semicolon")
        )
    }

    #[test]
    fn generated_header_requires_metadata_and_exact_source_agreement() {
        let header = render_declaration("entry", "float chelis_fn_656e747279(float x);");
        let parsed = GeneratedHeader::parse(&header).expect("generated header");
        let declaration = parsed.declaration("entry").expect("entry metadata");
        assert_eq!(declaration.symbol(), "chelis_fn_656e747279");
        parsed
            .validate_source(
                "/* chelis-authored-export: entry */\n\
                 float chelis_fn_656e747279(float x) {\n\
                 \x20   return x;\n\
                 }\n",
            )
            .expect("matching definition");

        assert!(GeneratedHeader::parse("float chelis_fn_656e747279(float x);").is_err());
        assert!(
            parsed
                .validate_source("float chelis_fn_656e747279(double x) {\n    return x;\n}\n")
                .is_err()
        );
    }

    #[test]
    fn generated_header_rejects_partial_extra_and_swapped_export_sets() {
        let alpha_decl = "int chelis_fn_616c706861(int x);";
        let beta_decl = "int chelis_fn_62657461(int x);";
        let complete_header = format!(
            "{}\n{}",
            render_declaration("alpha", alpha_decl),
            render_declaration("beta", beta_decl)
        );
        let complete_source = format!(
            "{}{}",
            authored_definition("alpha", alpha_decl, "return x + 1;"),
            authored_definition("beta", beta_decl, "return x + 100;")
        );
        let complete = GeneratedHeader::parse(&complete_header).expect("complete generated header");
        complete
            .validate_source(&complete_source)
            .expect("complete source/header association");

        let partial =
            GeneratedHeader::parse(&render_declaration("alpha", alpha_decl)).expect("partial");
        assert!(
            partial.validate_source(&complete_source).is_err(),
            "a header that omits an exported source definition must fail closed"
        );

        let extra_source = format!(
            "{complete_source}{}",
            authored_definition(
                "gamma",
                "int chelis_fn_67616d6d61(int x);",
                "return x + 1000;"
            )
        );
        assert!(
            complete.validate_source(&extra_source).is_err(),
            "an exported source definition absent from the header must fail closed"
        );
        let unmarked_extra_source = format!(
            "{complete_source}int chelis_fn_756e6d61726b6564(int x) {{\n    return x;\n}}\n"
        );
        assert!(
            complete.validate_source(&unmarked_extra_source).is_err(),
            "an unmarked external definition must not disappear from the exported set"
        );

        let swapped_header = format!(
            "{}\n{}",
            render_declaration("alpha", beta_decl),
            render_declaration("beta", alpha_decl)
        );
        let swapped =
            GeneratedHeader::parse(&swapped_header).expect("swapped metadata is syntactic");
        assert!(
            swapped.validate_source(&complete_source).is_err(),
            "source-name metadata must remain associated with its canonical emitted symbol"
        );

        let noncanonical_header = render_declaration("alpha", "int alpha(int x);");
        let noncanonical_source =
            authored_definition("alpha", "int alpha(int x);", "return x + 1;");
        assert!(
            GeneratedHeader::parse(&noncanonical_header)
                .expect("non-canonical header remains syntactic")
                .validate_source(&noncanonical_source)
                .is_err(),
            "authored exports must use the settled universal ABI, not a caller-selected identity"
        );
    }

    #[test]
    fn generated_header_exact_set_ignores_static_private_helpers() {
        let declaration = "int chelis_fn_616c706861(int x);";
        let header = render_declaration("alpha", declaration);
        let source = format!(
            "{}static int chelis_fn_616c706861__private(int x) {{\n    return x - 1;\n}}\n",
            authored_definition("alpha", declaration, "return x + 1;")
        );
        GeneratedHeader::parse(&header)
            .expect("generated header")
            .validate_source(&source)
            .expect("translation-unit-private helpers are not published exports");
    }
}
