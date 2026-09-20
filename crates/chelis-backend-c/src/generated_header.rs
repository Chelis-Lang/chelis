//! Generated C declaration metadata and source/header agreement checks.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

const HEADER_VERSION: &str = "/* chelis-generated-header: 1 */";
const SOURCE_VERSION: &str = "/* chelis-generated-source: 1 */";
const PROGRAM_IDENTITY_PREFIX: &str = "/* chelis-program-identity: ";
const SOURCE_DIGEST_PREFIX: &str = "/* chelis-source-sha256: ";
const DECLARATION_PREFIX: &str = "/* chelis-declaration: ";
const METADATA_SUFFIX: &str = " */";
const EXPORT_BEGIN_PREFIX: &str = "/* chelis-export-begin: ";
const EXPORT_END_PREFIX: &str = "/* chelis-export-end: ";

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
    program_identity: String,
    source_digest: String,
    declarations: BTreeMap<String, GeneratedDeclaration>,
}

impl GeneratedHeader {
    /// Parse a sealed generated C header and require its complete artifact envelope.
    pub fn parse(header: &str) -> Result<Self, GeneratedHeaderError> {
        let mut fields = header.splitn(4, '\n');
        if fields.next() != Some(HEADER_VERSION) {
            return Err(GeneratedHeaderError::new(
                "generated header is missing the artifact-version marker",
            ));
        }
        let program_identity = parse_hex_metadata(
            fields.next(),
            PROGRAM_IDENTITY_PREFIX,
            "generated program identity",
        )?;
        let source_digest = parse_digest_metadata(fields.next())?;
        let declaration_payload = fields.next().ok_or_else(|| {
            GeneratedHeaderError::new(
                "generated header is missing its declaration-payload separator",
            )
        })?;
        let declarations = parse_declarations(declaration_payload)?;
        Ok(Self {
            program_identity,
            source_digest,
            declarations,
        })
    }

    /// The exact compiler program/module identity bound into this artifact pair.
    pub fn program_identity(&self) -> &str {
        &self.program_identity
    }

    /// Return the declaration associated with one exact Chelis identity.
    pub fn declaration(&self, source_name: &str) -> Option<&GeneratedDeclaration> {
        self.declarations.get(source_name)
    }

    /// Require this exact generated source, program identity, and export manifest.
    pub fn validate_source(&self, source: &str) -> Result<(), GeneratedHeaderError> {
        let actual_digest = source_digest(source);
        if actual_digest != self.source_digest {
            return Err(GeneratedHeaderError::new(format!(
                "generated source digest disagrees with its header: expected {}, found {actual_digest}",
                self.source_digest
            )));
        }
        let (program_identity, definitions) = parse_source_manifest(source)?;
        if program_identity != self.program_identity {
            return Err(GeneratedHeaderError::new(format!(
                "generated source program identity `{program_identity}` disagrees with header identity `{}`",
                self.program_identity
            )));
        }
        validate_export_records(
            &self.program_identity,
            source,
            &self.declarations,
            &definitions,
        )
    }
}

fn parse_declarations(
    mut payload: &str,
) -> Result<BTreeMap<String, GeneratedDeclaration>, GeneratedHeaderError> {
    let mut declarations = BTreeMap::new();
    let mut record_number = 0;
    while !payload.is_empty() {
        record_number += 1;
        let (metadata, after_metadata) = payload.split_once('\n').ok_or_else(|| {
            GeneratedHeaderError::new(format!(
                "generated declaration record {record_number} has no C declaration payload"
            ))
        })?;
        let fields = metadata
            .strip_prefix(DECLARATION_PREFIX)
            .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
            .ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "generated declaration record {record_number} has invalid metadata"
                ))
            })?;
        let mut fields = fields.split_ascii_whitespace();
        let source_name = decode_hex_record_field(fields.next(), record_number, "source identity")?;
        let symbol = decode_hex_record_field(fields.next(), record_number, "canonical symbol")?;
        let declaration =
            decode_hex_record_field(fields.next(), record_number, "canonical declaration")?;
        if fields.next().is_some() {
            return Err(GeneratedHeaderError::new(format!(
                "generated declaration record {record_number} has extra metadata"
            )));
        }
        if !declaration.ends_with(';') {
            return Err(GeneratedHeaderError::new(format!(
                "generated declaration record {record_number} does not end in a semicolon"
            )));
        }
        let after_declaration = after_metadata
            .strip_prefix(&declaration)
            .ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "generated declaration record {record_number} metadata disagrees with its C declaration bytes"
                ))
            })?;
        payload = if after_declaration.is_empty() {
            ""
        } else {
            after_declaration.strip_prefix('\n').ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "generated declaration record {record_number} is not newline-delimited"
                ))
            })?
        };
        let declaration = GeneratedDeclaration {
            source_name: source_name.clone(),
            symbol,
            declaration,
        };
        if declarations
            .insert(source_name.clone(), declaration)
            .is_some()
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated declaration record {record_number} duplicates source identity `{source_name}`"
            )));
        }
    }
    Ok(declarations)
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

pub(crate) fn render_declaration(source_name: &str, symbol: &str, declaration: &str) -> String {
    format!(
        "{DECLARATION_PREFIX}{} {} {}{METADATA_SUFFIX}\n{declaration}",
        encode_hex(source_name),
        encode_hex(symbol),
        encode_hex(declaration)
    )
}

pub(crate) fn render_authored_export_begin(
    source_name: &str,
    symbol: &str,
    declaration: &str,
) -> String {
    render_export_begin(ExportKind::Authored, source_name, symbol, declaration)
}

pub(crate) fn render_direct_export_begin(
    source_name: &str,
    symbol: &str,
    declaration: &str,
) -> String {
    render_export_begin(ExportKind::Direct, source_name, symbol, declaration)
}

pub(crate) fn render_authored_export_end(source_name: &str) -> String {
    render_export_end(source_name)
}

pub(crate) fn render_direct_export_end(source_name: &str) -> String {
    render_export_end(source_name)
}

fn render_export_begin(
    kind: ExportKind,
    source_name: &str,
    symbol: &str,
    declaration: &str,
) -> String {
    format!(
        "{EXPORT_BEGIN_PREFIX}{} {} {} {}{METADATA_SUFFIX}",
        kind.as_str(),
        encode_hex(source_name),
        encode_hex(symbol),
        encode_hex(declaration)
    )
}

fn render_export_end(source_name: &str) -> String {
    format!(
        "{EXPORT_END_PREFIX}{}{METADATA_SUFFIX}",
        encode_hex(source_name)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportKind {
    Authored,
    Direct,
}

impl ExportKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Authored => "authored",
            Self::Direct => "direct",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "authored" => Some(Self::Authored),
            "direct" => Some(Self::Direct),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceDefinition {
    kind: ExportKind,
    source_name: String,
    symbol: String,
    declaration: String,
}

fn parse_source_manifest(
    source: &str,
) -> Result<(String, BTreeMap<String, SourceDefinition>), GeneratedHeaderError> {
    let mut lines = source.lines();
    if lines.next().map(str::trim) != Some(SOURCE_VERSION) {
        return Err(GeneratedHeaderError::new(
            "generated source is missing the artifact-version marker",
        ));
    }
    let program_identity = parse_hex_metadata(
        lines.next(),
        PROGRAM_IDENTITY_PREFIX,
        "generated program identity",
    )?;
    let mut definitions = BTreeMap::new();
    let mut pending_export: Option<SourceDefinition> = None;
    let mut payload_nonempty = false;

    for (index, raw_line) in lines.enumerate() {
        let line_number = index + 3;
        let line = raw_line.trim();
        if let Some(definition) = parse_export_begin(line, line_number)? {
            if let Some(pending) = pending_export.replace(definition) {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: export block for `{}` is not closed",
                    pending.source_name
                )));
            }
            payload_nonempty = false;
            continue;
        }
        if let Some(source_name) = parse_export_end(line)? {
            let definition = pending_export.take().ok_or_else(|| {
                GeneratedHeaderError::new(format!(
                    "line {line_number}: export end for `{source_name}` has no begin marker"
                ))
            })?;
            if definition.source_name != source_name {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: export block for `{}` closes as `{source_name}`",
                    definition.source_name
                )));
            }
            if !payload_nonempty {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: export block for `{source_name}` has no source definition"
                )));
            }
            if definitions
                .insert(source_name.clone(), definition)
                .is_some()
            {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: duplicate exported source definition for `{source_name}`"
                )));
            }
            continue;
        }
        if pending_export.is_some() && !line.is_empty() {
            payload_nonempty = true;
        }
    }

    if let Some(definition) = pending_export {
        return Err(GeneratedHeaderError::new(format!(
            "generated export block for `{}` has no end marker",
            definition.source_name
        )));
    }
    Ok((program_identity, definitions))
}

fn parse_export_begin(
    line: &str,
    line_number: usize,
) -> Result<Option<SourceDefinition>, GeneratedHeaderError> {
    let Some(payload) = line
        .strip_prefix(EXPORT_BEGIN_PREFIX)
        .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
    else {
        return Ok(None);
    };
    let mut fields = payload.split_ascii_whitespace();
    let kind = fields.next().and_then(ExportKind::parse).ok_or_else(|| {
        GeneratedHeaderError::new(format!(
            "line {line_number}: generated export block has an invalid kind"
        ))
    })?;
    let source_name = decode_hex_field(fields.next(), line_number, "source identity")?;
    let symbol = decode_hex_field(fields.next(), line_number, "canonical symbol")?;
    let declaration = decode_hex_field(fields.next(), line_number, "declaration")?;
    if fields.next().is_some() {
        return Err(GeneratedHeaderError::new(format!(
            "line {line_number}: generated export block has extra metadata"
        )));
    }
    Ok(Some(SourceDefinition {
        kind,
        source_name,
        symbol,
        declaration,
    }))
}

fn parse_export_end(line: &str) -> Result<Option<String>, GeneratedHeaderError> {
    line.strip_prefix(EXPORT_END_PREFIX)
        .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
        .map(decode_hex)
        .transpose()
}

fn validate_export_records(
    program_identity: &str,
    source: &str,
    declarations: &BTreeMap<String, GeneratedDeclaration>,
    definitions: &BTreeMap<String, SourceDefinition>,
) -> Result<(), GeneratedHeaderError> {
    for (source_name, declaration) in declarations {
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
        validate_export_symbol(program_identity, definition)?;
    }
    if let Some(source_name) = definitions
        .keys()
        .find(|source_name| !declarations.contains_key(*source_name))
    {
        return Err(GeneratedHeaderError::new(format!(
            "generated source exports `{source_name}` without a matching header declaration"
        )));
    }
    reject_export_macro_aliases(
        source,
        definitions.values().map(|item| item.symbol.as_str()),
    )
}

fn validate_export_symbol(
    program_identity: &str,
    definition: &SourceDefinition,
) -> Result<(), GeneratedHeaderError> {
    let valid = match definition.kind {
        ExportKind::Direct => definition.symbol == definition.source_name,
        ExportKind::Authored if definition.source_name == "main" => {
            definition.symbol == format!("{program_identity}__main")
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

pub(crate) fn seal_generated_artifact(
    program_identity: &str,
    source: &str,
    header: &str,
) -> Result<(String, String), GeneratedHeaderError> {
    if program_identity.is_empty() {
        return Err(GeneratedHeaderError::new(
            "generated program identity is empty",
        ));
    }
    let raw_source = strip_source_envelope(source)?;
    let raw_header = strip_header_envelope(header)?;
    let declarations = parse_declarations(raw_header)?;
    let sealed_source = format!(
        "{SOURCE_VERSION}\n{PROGRAM_IDENTITY_PREFIX}{}{METADATA_SUFFIX}\n{raw_source}",
        encode_hex(program_identity)
    );
    let (source_identity, definitions) = parse_source_manifest(&sealed_source)?;
    debug_assert_eq!(source_identity, program_identity);
    validate_export_records(
        program_identity,
        &sealed_source,
        &declarations,
        &definitions,
    )?;
    let digest = source_digest(&sealed_source);
    let sealed_header = format!(
        "{HEADER_VERSION}\n{PROGRAM_IDENTITY_PREFIX}{}{METADATA_SUFFIX}\n{SOURCE_DIGEST_PREFIX}{digest}{METADATA_SUFFIX}\n{raw_header}",
        encode_hex(program_identity)
    );
    Ok((sealed_source, sealed_header))
}

fn strip_source_envelope(source: &str) -> Result<&str, GeneratedHeaderError> {
    if !source.starts_with(SOURCE_VERSION) {
        return Ok(source);
    }
    let (_, after_version) = source.split_once('\n').ok_or_else(|| {
        GeneratedHeaderError::new("generated source envelope has no program identity")
    })?;
    let (_, raw_source) = after_version
        .split_once('\n')
        .ok_or_else(|| GeneratedHeaderError::new("generated source envelope has no C payload"))?;
    Ok(raw_source)
}

fn strip_header_envelope(header: &str) -> Result<&str, GeneratedHeaderError> {
    if !header.starts_with(HEADER_VERSION) {
        return Ok(header);
    }
    let mut offset = 0;
    for _ in 0..3 {
        let remaining = &header[offset..];
        let newline = remaining.find('\n').ok_or_else(|| {
            GeneratedHeaderError::new("generated header envelope has no declaration payload")
        })?;
        offset += newline + 1;
    }
    Ok(&header[offset..])
}

fn parse_hex_metadata(
    line: Option<&str>,
    prefix: &str,
    label: &str,
) -> Result<String, GeneratedHeaderError> {
    let line = line.map(str::trim).ok_or_else(|| {
        GeneratedHeaderError::new(format!("generated artifact is missing {label}"))
    })?;
    let encoded = line
        .strip_prefix(prefix)
        .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
        .ok_or_else(|| {
            GeneratedHeaderError::new(format!("generated artifact has invalid {label} metadata"))
        })?;
    decode_hex(encoded)
}

fn parse_digest_metadata(line: Option<&str>) -> Result<String, GeneratedHeaderError> {
    let line = line.map(str::trim).ok_or_else(|| {
        GeneratedHeaderError::new("generated header is missing its source digest")
    })?;
    let digest = line
        .strip_prefix(SOURCE_DIGEST_PREFIX)
        .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
        .ok_or_else(|| {
            GeneratedHeaderError::new("generated header has invalid source digest metadata")
        })?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(GeneratedHeaderError::new(
            "generated header source digest is not canonical lowercase SHA-256",
        ));
    }
    Ok(digest.to_string())
}

fn source_digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

fn encode_hex(value: &str) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.bytes() {
        write!(encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn decode_hex_field(
    field: Option<&str>,
    line_number: usize,
    label: &str,
) -> Result<String, GeneratedHeaderError> {
    let field = field.ok_or_else(|| {
        GeneratedHeaderError::new(format!(
            "line {line_number}: generated export block is missing {label}"
        ))
    })?;
    decode_hex(field)
}

fn decode_hex_record_field(
    field: Option<&str>,
    record_number: usize,
    label: &str,
) -> Result<String, GeneratedHeaderError> {
    let field = field.ok_or_else(|| {
        GeneratedHeaderError::new(format!(
            "generated declaration record {record_number} is missing {label}"
        ))
    })?;
    decode_hex(field)
}

fn decode_hex(value: &str) -> Result<String, GeneratedHeaderError> {
    if value.is_empty()
        || !value.len().is_multiple_of(2)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(GeneratedHeaderError::new(
            "generated artifact contains invalid hexadecimal metadata",
        ));
    }
    let encoded = value.as_bytes();
    let bytes = (0..encoded.len())
        .step_by(2)
        .map(|index| {
            let pair = &encoded[index..index + 2];
            let pair = std::str::from_utf8(pair).expect("hexadecimal metadata is ASCII");
            u8::from_str_radix(pair, 16).expect("validated hexadecimal pair")
        })
        .collect::<Vec<_>>();
    String::from_utf8(bytes)
        .map_err(|_| GeneratedHeaderError::new("generated artifact metadata is not UTF-8"))
}

fn reject_export_macro_aliases<'a>(
    source: &str,
    symbols: impl Iterator<Item = &'a str>,
) -> Result<(), GeneratedHeaderError> {
    let symbols = symbols.collect::<std::collections::BTreeSet<_>>();
    for logical_line in preprocessor_logical_lines(source) {
        let Some(name) = macro_definition_name(&logical_line) else {
            continue;
        };
        if symbols.contains(name.as_str()) {
            return Err(GeneratedHeaderError::new(format!(
                "generated source preprocessor-rebinds exported symbol `{name}`"
            )));
        }
    }
    Ok(())
}

fn preprocessor_logical_lines(source: &str) -> Vec<String> {
    strip_c_comments(&splice_c_lines(source))
        .lines()
        .map(str::to_string)
        .collect()
}

fn splice_c_lines(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut spliced = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            if bytes.get(index + 1) == Some(&b'\n') {
                index += 2;
                continue;
            }
            if bytes.get(index + 1) == Some(&b'\r') && bytes.get(index + 2) == Some(&b'\n') {
                index += 3;
                continue;
            }
        }
        spliced.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(spliced).expect("splicing UTF-8 at ASCII line boundaries preserves UTF-8")
}

fn strip_c_comments(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        BlockComment,
        LineComment,
        Quoted(u8),
    }

    let bytes = source.as_bytes();
    let mut stripped = Vec::with_capacity(bytes.len());
    let mut state = State::Normal;
    let mut index = 0;
    while index < bytes.len() {
        match state {
            State::Normal if bytes[index..].starts_with(b"/*") => {
                stripped.push(b' ');
                state = State::BlockComment;
                index += 2;
            }
            State::Normal if bytes[index..].starts_with(b"//") => {
                stripped.push(b' ');
                state = State::LineComment;
                index += 2;
            }
            State::Normal if matches!(bytes[index], b'"' | b'\'') => {
                stripped.push(bytes[index]);
                state = State::Quoted(bytes[index]);
                index += 1;
            }
            State::Normal => {
                stripped.push(bytes[index]);
                index += 1;
            }
            State::BlockComment if bytes[index..].starts_with(b"*/") => {
                state = State::Normal;
                index += 2;
            }
            State::BlockComment => {
                if bytes[index] == b'\n' {
                    stripped.push(b'\n');
                }
                index += 1;
            }
            State::LineComment => {
                if bytes[index] == b'\n' {
                    stripped.push(b'\n');
                    state = State::Normal;
                }
                index += 1;
            }
            State::Quoted(_) if bytes[index] == b'\\' => {
                stripped.push(bytes[index]);
                index += 1;
                if index < bytes.len() {
                    stripped.push(bytes[index]);
                    index += 1;
                }
            }
            State::Quoted(quote) if bytes[index] == quote => {
                stripped.push(bytes[index]);
                state = State::Normal;
                index += 1;
            }
            State::Quoted(_) => {
                stripped.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8(stripped).expect("comment stripping preserves UTF-8 source bytes")
}

fn macro_definition_name(line: &str) -> Option<String> {
    let mut cursor = CDirectiveCursor::new(line);
    cursor.skip_trivia()?;
    cursor.take_directive_introducer()?;
    cursor.skip_trivia()?;
    let directive = cursor.identifier()?;
    if directive != "define" && directive != "undef" {
        return None;
    }
    cursor.skip_trivia()?;
    cursor.identifier()
}

struct CDirectiveCursor<'a> {
    remaining: &'a str,
}

impl<'a> CDirectiveCursor<'a> {
    fn new(remaining: &'a str) -> Self {
        Self { remaining }
    }

    fn skip_trivia(&mut self) -> Option<()> {
        loop {
            self.remaining = self.remaining.trim_start();
            if let Some(rest) = self.remaining.strip_prefix("/*") {
                let end = rest.find("*/")?;
                self.remaining = &rest[end + 2..];
                continue;
            }
            return Some(());
        }
    }

    fn take_directive_introducer(&mut self) -> Option<()> {
        self.remaining = self
            .remaining
            .strip_prefix('#')
            .or_else(|| self.remaining.strip_prefix("%:"))
            .or_else(|| self.remaining.strip_prefix("??="))?;
        Some(())
    }

    fn identifier(&mut self) -> Option<String> {
        let mut chars = self.remaining.char_indices();
        let (_, first) = chars.next()?;
        if first != '_' && !first.is_ascii_alphabetic() {
            return None;
        }
        let mut end = first.len_utf8();
        for (index, ch) in chars {
            if ch != '_' && !ch.is_ascii_alphanumeric() {
                break;
            }
            end = index + ch.len_utf8();
        }
        let identifier = self.remaining[..end].to_string();
        self.remaining = &self.remaining[end..];
        Some(identifier)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GeneratedHeader, encode_hex, render_authored_export_begin, render_authored_export_end,
        render_declaration, seal_generated_artifact,
    };

    fn declaration_record_prefix(source_name: &str) -> String {
        format!("/* chelis-declaration: {} ", encode_hex(source_name))
    }

    #[test]
    fn generated_header_requires_metadata_and_exact_source_agreement() {
        let (source, header) = sealed_fixture(
            "demo",
            "entry",
            "chelis_fn_656e747279",
            "float chelis_fn_656e747279(float x);",
            "float chelis_fn_656e747279(float x) {\n    return x;\n}",
            "",
        );
        let parsed = GeneratedHeader::parse(&header).expect("generated header");
        let declaration = parsed.declaration("entry").expect("entry metadata");
        assert_eq!(declaration.symbol(), "chelis_fn_656e747279");
        parsed
            .validate_source(&source)
            .expect("matching definition");

        assert!(GeneratedHeader::parse("float chelis_fn_656e747279(float x);").is_err());
        assert!(
            GeneratedHeader::parse(&header.replacen("64656d6f", "64656D6F", 1)).is_err(),
            "uppercase hexadecimal metadata is not canonical"
        );
        assert!(
            parsed
                .validate_source(&source.replace("(float x)", "(double x)"))
                .is_err()
        );
    }

    #[test]
    fn generated_header_rejects_partial_extra_and_swapped_export_sets() {
        let alpha_decl = "int chelis_fn_616c706861(int x);";
        let beta_decl = "int chelis_fn_62657461(int x);";
        let complete_header = format!(
            "{}\n{}",
            render_declaration("alpha", "chelis_fn_616c706861", alpha_decl),
            render_declaration("beta", "chelis_fn_62657461", beta_decl)
        );
        let (_, alpha_source) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            alpha_decl,
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let (_, beta_source) = raw_export(
            "beta",
            "chelis_fn_62657461",
            beta_decl,
            "int chelis_fn_62657461(int x) {\n    return x + 100;\n}",
        );
        let (complete_source, complete_header) = seal_generated_artifact(
            "demo",
            &format!("{alpha_source}\n{beta_source}\n"),
            &complete_header,
        )
        .expect("complete generated artifact");
        let complete = GeneratedHeader::parse(&complete_header).expect("complete generated header");
        complete
            .validate_source(&complete_source)
            .expect("complete source/header association");

        let partial_header = complete_header.replace(
            &render_declaration("beta", "chelis_fn_62657461", beta_decl),
            "",
        );
        let partial = GeneratedHeader::parse(&partial_header).expect("partial");
        assert!(
            partial.validate_source(&complete_source).is_err(),
            "a header that omits an exported source definition must fail closed"
        );

        let (_, gamma_source) = raw_export(
            "gamma",
            "chelis_fn_67616d6d61",
            "int chelis_fn_67616d6d61(int x);",
            "int chelis_fn_67616d6d61(int x) {\n    return x + 1000;\n}",
        );
        let extra_source = format!("{complete_source}{gamma_source}\n");
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

        let swapped_header = complete_header
            .replacen(
                &declaration_record_prefix("alpha"),
                &declaration_record_prefix("__swap"),
                1,
            )
            .replacen(
                &declaration_record_prefix("beta"),
                &declaration_record_prefix("alpha"),
                1,
            )
            .replacen(
                &declaration_record_prefix("__swap"),
                &declaration_record_prefix("beta"),
                1,
            );
        let swapped =
            GeneratedHeader::parse(&swapped_header).expect("swapped metadata is syntactic");
        assert!(
            swapped.validate_source(&complete_source).is_err(),
            "source-name metadata must remain associated with its canonical emitted symbol"
        );

        let (noncanonical_header, noncanonical_source) = raw_export(
            "alpha",
            "alpha",
            "int alpha(int x);",
            "int alpha(int x) {\n    return x + 1;\n}",
        );
        assert!(
            seal_generated_artifact("demo", &noncanonical_source, &noncanonical_header).is_err(),
            "authored exports must use the settled universal ABI, not a caller-selected identity"
        );
    }

    #[test]
    fn generated_header_exact_set_ignores_static_private_helpers() {
        let declaration = "int chelis_fn_616c706861(int x);";
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            declaration,
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
            "static int chelis_fn_616c706861__private(int x) {\n    return x - 1;\n}\n",
        );
        GeneratedHeader::parse(&header)
            .expect("generated header")
            .validate_source(&source)
            .expect("translation-unit-private helpers are not published exports");
    }

    fn raw_export(
        source_name: &str,
        symbol: &str,
        declaration: &str,
        definition: &str,
    ) -> (String, String) {
        (
            render_declaration(source_name, symbol, declaration),
            format!(
                "{}\n{definition}\n{}",
                render_authored_export_begin(source_name, symbol, declaration),
                render_authored_export_end(source_name)
            ),
        )
    }

    fn sealed_fixture(
        program_identity: &str,
        source_name: &str,
        symbol: &str,
        declaration: &str,
        definition: &str,
        private_source: &str,
    ) -> (String, String) {
        let (header, export) = raw_export(source_name, symbol, declaration, definition);
        seal_generated_artifact(
            program_identity,
            &format!("{private_source}{export}\n"),
            &header,
        )
        .expect("valid generated artifact")
    }

    #[test]
    fn sealed_artifact_accepts_multiline_definition_and_comment_separated_static_helper() {
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            "int\nchelis_fn_616c706861(\n    int x\n);",
            "int\nchelis_fn_616c706861\n(\n    int x\n)\n{\n    return x + 1;\n}",
            "static/* generated helper */ inline int private_helper(int x) {\n    return x;\n}\n",
        );
        GeneratedHeader::parse(&header)
            .expect("sealed header")
            .validate_source(&source)
            .expect("C formatting inside an export block is opaque to artifact validation");
    }

    #[test]
    fn sealed_artifact_rejects_multiline_unmarked_and_comment_spoofed_external_definitions() {
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
            "",
        );
        let generated = GeneratedHeader::parse(&header).expect("sealed header");
        let multiline = format!("{source}\nint unmarked_extra(int x)\n{{\n    return x - 1;\n}}\n");
        assert!(
            generated.validate_source(&multiline).is_err(),
            "whole-source binding must reject an unmarked multiline external definition"
        );
        let comment_spoofed =
            format!("{source}\nint /* static */ unmarked_extra(int x) {{\n    return x - 1;\n}}\n");
        assert!(
            generated.validate_source(&comment_spoofed).is_err(),
            "a comment containing `static` must not hide a changed generated source"
        );
    }

    #[test]
    fn sealed_artifact_rejects_export_symbol_macro_aliases() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let source = format!(
            "#define chelis_fn_616c706861 chelis_fn_62657461\n{export}\n#undef chelis_fn_616c706861\n"
        );
        assert!(
            seal_generated_artifact("demo", &source, &header).is_err(),
            "a compiler artifact may not macro-rebind a published symbol"
        );

        for disguised_directive in [
            "/* leading\ncomment */ %:define chelis_fn_616c706861 chelis_fn_62657461",
            "#/* separator */def\\\nine chelis_fn_616c706861 chelis_fn_62657461",
            "??=define chelis_fn_616c706861 chelis_fn_62657461",
        ] {
            let source = format!("{disguised_directive}\n{export}\n");
            assert!(
                seal_generated_artifact("demo", &source, &header).is_err(),
                "preprocessing-token spelling `{disguised_directive}` must not hide an export alias"
            );
        }
    }

    #[test]
    fn unqualified_source_main_is_bound_to_exact_program_identity() {
        let (bad_header, bad_export) = raw_export(
            "main",
            "attacker__main",
            "int attacker__main(int x);",
            "int attacker__main(int x) {\n    return x;\n}",
        );
        assert!(
            seal_generated_artifact("demo", &bad_export, &bad_header).is_err(),
            "suffix resemblance is not the program-derived source-main ABI"
        );

        let (source, header) = sealed_fixture(
            "demo",
            "main",
            "demo__main",
            "int demo__main(int x);",
            "int demo__main(int x) {\n    return x;\n}",
            "",
        );
        GeneratedHeader::parse(&header)
            .expect("sealed header")
            .validate_source(&source)
            .expect("the exact program-derived source-main symbol remains valid");
    }
}
