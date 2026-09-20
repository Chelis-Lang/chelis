//! Generated C declaration metadata and source/header agreement checks.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

const HEADER_VERSION: &str = "/* chelis-generated-header: 2 */";
const SOURCE_VERSION: &str = "/* chelis-generated-source: 2 */";
const PROGRAM_IDENTITY_PREFIX: &str = "/* chelis-program-identity: ";
const SOURCE_DIGEST_PREFIX: &str = "/* chelis-source-sha256: ";
const DECLARATION_PREFIX: &str = "/* chelis-declaration: ";
const METADATA_SUFFIX: &str = " */";
const EXPORT_BEGIN_PREFIX: &str = "/* chelis-export-begin: ";
const EXPORT_END_PREFIX: &str = "/* chelis-export-end: ";
const PLACEHOLDER_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestMode {
    Raw,
    Sealed,
}

/// One public function declaration from a generated C header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedDeclaration {
    source_name: String,
    symbol: String,
    declaration: String,
    definition_digest: String,
    record_number: usize,
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

    /// SHA-256 commitment to the exact definition bytes and source-local
    /// preprocessing context.
    pub fn definition_digest(&self) -> &str {
        &self.definition_digest
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
        let declarations = parse_declarations(declaration_payload, ManifestMode::Sealed)?;
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
        self.validate_structure(source)
    }

    fn validate_structure(&self, source: &str) -> Result<(), GeneratedHeaderError> {
        let (program_identity, definitions) = parse_source_manifest(source, ManifestMode::Sealed)?;
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
    mode: ManifestMode,
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
        let definition_digest = match mode {
            ManifestMode::Raw => {
                if fields.next().is_some() {
                    return Err(GeneratedHeaderError::new(format!(
                        "generated declaration record {record_number} has extra metadata"
                    )));
                }
                String::new()
            }
            ManifestMode::Sealed => {
                let digest = parse_record_digest(
                    fields.next(),
                    &format!("generated declaration record {record_number}"),
                )?;
                if fields.next().is_some() {
                    return Err(GeneratedHeaderError::new(format!(
                        "generated declaration record {record_number} has extra metadata"
                    )));
                }
                digest
            }
        };
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
            definition_digest,
            record_number,
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
    render_declaration_record(source_name, symbol, declaration, None)
}

fn render_declaration_record(
    source_name: &str,
    symbol: &str,
    declaration: &str,
    definition_digest: Option<&str>,
) -> String {
    let digest = if let Some(digest) = definition_digest {
        format!(" {digest}")
    } else {
        String::new()
    };
    format!(
        "{DECLARATION_PREFIX}{} {} {}{digest}{METADATA_SUFFIX}\n{declaration}",
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
    render_export_begin_record(kind, source_name, symbol, declaration, None)
}

fn render_export_begin_record(
    kind: ExportKind,
    source_name: &str,
    symbol: &str,
    declaration: &str,
    definition_digest: Option<&str>,
) -> String {
    let digest = if let Some(digest) = definition_digest {
        format!(" {digest}")
    } else {
        String::new()
    };
    format!(
        "{EXPORT_BEGIN_PREFIX}{} {} {} {}{digest}{METADATA_SUFFIX}",
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
    definition_digest: String,
    payload_start: usize,
    payload_end: usize,
    marker_start: usize,
    marker_end: usize,
}

fn parse_source_manifest(
    source: &str,
    mode: ManifestMode,
) -> Result<(String, BTreeMap<String, SourceDefinition>), GeneratedHeaderError> {
    let mut offset = 0;
    let mut lines = source.split_inclusive('\n');
    let first_raw = lines.next();
    let first = first_raw.map(trim_line_ending);
    if first.map(str::trim) != Some(SOURCE_VERSION) {
        return Err(GeneratedHeaderError::new(
            "generated source is missing the artifact-version marker",
        ));
    }
    offset += match first_raw {
        Some(line) => line.len(),
        None => 0,
    };
    let identity_raw = lines.next();
    let identity_line = identity_raw.map(trim_line_ending);
    let program_identity = parse_hex_metadata(
        identity_line,
        PROGRAM_IDENTITY_PREFIX,
        "generated program identity",
    )?;
    offset += match identity_raw {
        Some(line) => line.len(),
        None => 0,
    };
    let mut definitions = BTreeMap::new();
    let mut pending_export: Option<SourceDefinition> = None;
    let mut payload_nonempty = false;

    for (index, raw_line_with_ending) in lines.enumerate() {
        let line_number = index + 3;
        let raw_line = trim_line_ending(raw_line_with_ending);
        let line = raw_line.trim();
        if let Some(mut definition) = parse_export_begin(line, line_number, mode)? {
            if let Some(pending) = &pending_export {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: export block for `{}` is not closed",
                    pending.source_name
                )));
            }
            definition.marker_start = offset;
            definition.marker_end = offset + raw_line.len();
            definition.payload_start = offset + raw_line_with_ending.len();
            pending_export = Some(definition);
            payload_nonempty = false;
            offset += raw_line_with_ending.len();
            continue;
        }
        if let Some(source_name) = parse_export_end(line)? {
            let mut definition = pending_export.take().ok_or_else(|| {
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
            definition.payload_end = offset;
            if definitions
                .insert(source_name.clone(), definition)
                .is_some()
            {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: duplicate exported source definition for `{source_name}`"
                )));
            }
            offset += raw_line_with_ending.len();
            continue;
        }
        if pending_export.is_some() && !line.is_empty() {
            payload_nonempty = true;
        }
        offset += raw_line_with_ending.len();
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
    mode: ManifestMode,
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
    let definition_digest = match mode {
        ManifestMode::Raw => {
            if fields.next().is_some() {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: generated export block has extra metadata"
                )));
            }
            String::new()
        }
        ManifestMode::Sealed => {
            let digest = parse_record_digest(
                fields.next(),
                &format!("line {line_number}: generated export block"),
            )?;
            if fields.next().is_some() {
                return Err(GeneratedHeaderError::new(format!(
                    "line {line_number}: generated export block has extra metadata"
                )));
            }
            digest
        }
    };
    Ok(Some(SourceDefinition {
        kind,
        source_name,
        symbol,
        declaration,
        definition_digest,
        payload_start: 0,
        payload_end: 0,
        marker_start: 0,
        marker_end: 0,
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
    validate_export_identities(program_identity, declarations, definitions)?;
    reject_export_macro_aliases(source, definitions.values())?;
    validate_c_definitions(source, declarations, definitions)
}

fn validate_export_identities(
    program_identity: &str,
    declarations: &BTreeMap<String, GeneratedDeclaration>,
    definitions: &BTreeMap<String, SourceDefinition>,
) -> Result<(), GeneratedHeaderError> {
    let mut symbol_owners = BTreeMap::new();
    for (source_name, declaration) in declarations {
        let definition = definitions.get(source_name).ok_or_else(|| {
            GeneratedHeaderError::new(format!(
                "generated source has no exported definition for `{source_name}`"
            ))
        })?;
        if declaration.symbol != definition.symbol
            || declaration.declaration != definition.declaration
            || declaration.definition_digest != definition.definition_digest
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated header/source association disagrees for `{source_name}`: header `{}`, source `{}`",
                declaration.declaration, definition.declaration
            )));
        }
        validate_export_symbol(program_identity, definition)?;
        if let Some(existing_source_name) =
            symbol_owners.insert(definition.symbol.as_str(), source_name.as_str())
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated exports `{existing_source_name}` and `{source_name}` claim the same external C symbol `{}`",
                definition.symbol
            )));
        }
    }
    if let Some(source_name) = definitions
        .keys()
        .find(|source_name| !declarations.contains_key(*source_name))
    {
        return Err(GeneratedHeaderError::new(format!(
            "generated source exports `{source_name}` without a matching header declaration"
        )));
    }
    Ok(())
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

#[derive(Debug)]
struct CFunctionDefinition {
    start: usize,
    end: usize,
    symbol: String,
    is_static: bool,
    signature_tokens: Vec<String>,
}

struct ParsedCTranslationUnit {
    functions: Vec<CFunctionDefinition>,
    error_context: String,
}

fn validate_c_definitions(
    source: &str,
    declarations: &BTreeMap<String, GeneratedDeclaration>,
    definitions: &BTreeMap<String, SourceDefinition>,
) -> Result<(), GeneratedHeaderError> {
    let digests = bind_c_definitions(source, definitions, true)?;
    for (source_name, declaration) in declarations {
        let actual = digests
            .get(source_name)
            .expect("the exact export set was already validated");
        if actual != &declaration.definition_digest {
            return Err(GeneratedHeaderError::new(format!(
                "generated declaration for `{source_name}` does not commit to its exact C definition"
            )));
        }
    }
    Ok(())
}

fn bind_c_definitions(
    source: &str,
    definitions: &BTreeMap<String, SourceDefinition>,
    enforce_digest: bool,
) -> Result<BTreeMap<String, String>, GeneratedHeaderError> {
    validate_generated_include_set(source)?;
    validate_generated_pragmas(source)?;
    if preprocessor_conditional_depth(source)? != 0 {
        return Err(GeneratedHeaderError::new(
            "generated source has an unclosed preprocessor conditional",
        ));
    }
    let parsed = parse_c_function_definitions(source)?;
    let functions = &parsed.functions;
    let mut claimed = vec![false; functions.len()];
    let mut digests = BTreeMap::new();

    for (source_name, definition) in definitions {
        let enclosed = functions
            .iter()
            .enumerate()
            .filter(|(_, function)| {
                function.start >= definition.payload_start && function.end <= definition.payload_end
            })
            .collect::<Vec<_>>();
        if enclosed.len() != 1 {
            return Err(GeneratedHeaderError::new(format!(
                "generated export block for `{source_name}` encloses {} C function definitions instead of exactly one",
                enclosed.len()
            )));
        }
        let (index, function) = enclosed[0];
        if preprocessor_conditional_depth(&source[..function.start])? != 0
            || preprocessor_conditional_depth(&source[..function.end])? != 0
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated export `{source_name}` is conditionally included"
            )));
        }
        claimed[index] = true;
        if function.is_static {
            return Err(GeneratedHeaderError::new(format!(
                "generated export `{source_name}` is defined with internal linkage"
            )));
        }
        if function.symbol != definition.symbol {
            return Err(GeneratedHeaderError::new(format!(
                "generated export `{source_name}` records symbol `{}` but defines `{}`",
                definition.symbol, function.symbol
            )));
        }
        let declaration_tokens = parse_c_declaration_tokens(&definition.declaration)?;
        if function.signature_tokens != declaration_tokens {
            return Err(GeneratedHeaderError::new(format!(
                "generated export `{source_name}` declaration does not structurally match its C definition"
            )));
        }
        let digest = definition_commitment(source, function, &parsed.error_context);
        if enforce_digest && digest != definition.definition_digest {
            return Err(GeneratedHeaderError::new(format!(
                "generated export `{source_name}` definition digest disagrees with its exact C definition"
            )));
        }
        digests.insert(source_name.clone(), digest);
    }

    let mut process_main_count = 0;
    for (index, function) in functions.iter().enumerate() {
        if claimed[index] || function.is_static {
            continue;
        }
        if function.symbol == "main" {
            process_main_count += 1;
            continue;
        }
        return Err(GeneratedHeaderError::new(format!(
            "generated source contains unregistered externally linked function definition `{}`",
            function.symbol
        )));
    }
    if process_main_count > 1 {
        return Err(GeneratedHeaderError::new(
            "generated source contains more than one process entry `main`",
        ));
    }
    Ok(digests)
}

fn definition_commitment(
    source: &str,
    function: &CFunctionDefinition,
    error_context: &str,
) -> String {
    let directives = preprocessor_logical_lines(source)
        .into_iter()
        .filter(|line| is_preprocessor_directive(line));
    let mut hasher = Sha256::new();
    hasher.update(b"chelis-c-definition-v2\0");
    for directive in directives {
        hasher.update(directive.len().to_le_bytes());
        hasher.update(directive.as_bytes());
    }
    hasher.update(b"\0parser-errors\0");
    hasher.update(error_context.as_bytes());
    hasher.update(b"\0definition\0");
    hasher.update(&source.as_bytes()[function.start..function.end]);
    format!("{:x}", hasher.finalize())
}

fn parse_c_function_definitions(
    source: &str,
) -> Result<ParsedCTranslationUnit, GeneratedHeaderError> {
    let projected = normalize_manifest_digests_for_c_parse(source);
    let projected_source = projected.as_str();
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_c::LANGUAGE.into();
    parser.set_language(&language).map_err(|error| {
        GeneratedHeaderError::new(format!("cannot initialize generated C parser: {error}"))
    })?;
    let tree = parser
        .parse(projected_source, None)
        .ok_or_else(|| GeneratedHeaderError::new("generated C parser returned no syntax tree"))?;
    if tree.root_node().has_error() {
        return Err(GeneratedHeaderError::new(
            "generated C source is not structurally parseable before preprocessing",
        ));
    }
    let mut nodes = Vec::new();
    collect_function_nodes(tree.root_node(), &mut nodes);
    let functions = nodes
        .into_iter()
        .map(|node| {
            let declarator = node.child_by_field_name("declarator").ok_or_else(|| {
                GeneratedHeaderError::new("generated C function has no declarator")
            })?;
            let symbol = declarator_identifier(declarator, projected_source.as_bytes())
                .ok_or_else(|| {
                    GeneratedHeaderError::new(
                        "generated C function declarator has no exact identifier",
                    )
                })?;
            Ok(CFunctionDefinition {
                start: node.start_byte(),
                end: node.end_byte(),
                symbol,
                is_static: has_static_function_storage(node, projected_source.as_bytes()),
                signature_tokens: function_signature_tokens(node, projected_source.as_bytes()),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut errors = Vec::new();
    collect_c_errors(tree.root_node(), &mut errors);
    let mut error_hasher = Sha256::new();
    for node in errors {
        error_hasher.update(node.kind().as_bytes());
        error_hasher.update(&projected_source.as_bytes()[node.start_byte()..node.end_byte()]);
    }
    Ok(ParsedCTranslationUnit {
        functions,
        error_context: format!("{:x}", error_hasher.finalize()),
    })
}

fn normalize_manifest_digests_for_c_parse(source: &str) -> String {
    let mut projected = source.to_string();
    let mut offset = 0;
    for line_with_ending in source.split_inclusive('\n') {
        let line = trim_line_ending(line_with_ending);
        let trimmed = line.trim();
        if let Some(payload) = trimmed
            .strip_prefix(EXPORT_BEGIN_PREFIX)
            .and_then(|line| line.strip_suffix(METADATA_SUFFIX))
            && let Some(digest) = payload.split_ascii_whitespace().nth(4)
            && digest.len() == PLACEHOLDER_DIGEST.len()
        {
            let line_digest_start = line
                .rfind(digest)
                .expect("parsed export digest occurs in its marker");
            let start = offset + line_digest_start;
            projected.replace_range(start..start + digest.len(), PLACEHOLDER_DIGEST);
        }
        offset += line_with_ending.len();
    }
    mask_cxx_linkage_specs_for_c_parse(source, &mut projected);
    projected
}

fn mask_cxx_linkage_specs_for_c_parse(source: &str, projected: &mut String) {
    let mut offset = 0;
    for line_with_ending in source.split_inclusive('\n') {
        let line = trim_line_ending(line_with_ending);
        if line.trim() == "extern \"C\"" {
            projected.replace_range(offset..offset + line.len(), &" ".repeat(line.len()));
        }
        offset += line_with_ending.len();
    }
}

fn collect_c_errors<'tree>(
    node: tree_sitter::Node<'tree>,
    out: &mut Vec<tree_sitter::Node<'tree>>,
) {
    if node.is_error() || node.is_missing() {
        out.push(node);
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_c_errors(child, out);
    }
}

fn collect_function_nodes<'tree>(
    node: tree_sitter::Node<'tree>,
    out: &mut Vec<tree_sitter::Node<'tree>>,
) {
    if node.kind() == "function_definition" {
        out.push(node);
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_function_nodes(child, out);
    }
}

fn declarator_identifier(node: tree_sitter::Node<'_>, source: &[u8]) -> Option<String> {
    if node.kind() == "identifier" {
        return node.utf8_text(source).ok().map(str::to_string);
    }
    if let Some(inner) = node.child_by_field_name("declarator")
        && let Some(identifier) = declarator_identifier(inner, source)
    {
        return Some(identifier);
    }
    None
}

fn has_static_function_storage(node: tree_sitter::Node<'_>, source: &[u8]) -> bool {
    let Some(declarator) = node.child_by_field_name("declarator") else {
        return false;
    };
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .take_while(|child| child.end_byte() <= declarator.start_byte())
        .any(|child| {
            child.kind() == "storage_class_specifier"
                && child.utf8_text(source).is_ok_and(|text| text == "static")
        })
}

fn function_signature_tokens(node: tree_sitter::Node<'_>, source: &[u8]) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "compound_statement" {
            collect_c_tokens(child, source, &mut tokens);
        }
    }
    tokens
}

fn parse_c_declaration_tokens(declaration: &str) -> Result<Vec<String>, GeneratedHeaderError> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_c::LANGUAGE.into();
    parser.set_language(&language).map_err(|error| {
        GeneratedHeaderError::new(format!("cannot initialize generated C parser: {error}"))
    })?;
    let tree = parser.parse(declaration, None).ok_or_else(|| {
        GeneratedHeaderError::new("generated C declaration parser returned no syntax tree")
    })?;
    let root = tree.root_node();
    if root.has_error() || root.named_child_count() != 1 {
        return Err(GeneratedHeaderError::new(
            "generated C declaration is not structurally parseable",
        ));
    }
    let declaration_node = root
        .named_child(0)
        .ok_or_else(|| GeneratedHeaderError::new("generated C declaration has no syntax node"))?;
    if declaration_node.kind() != "declaration" {
        return Err(GeneratedHeaderError::new(
            "generated C declaration metadata is not a function declaration",
        ));
    }
    let mut tokens = Vec::new();
    collect_c_tokens(declaration_node, declaration.as_bytes(), &mut tokens);
    if tokens.last().is_some_and(|token| token == ";") {
        tokens.pop();
    }
    Ok(tokens)
}

fn collect_c_tokens(node: tree_sitter::Node<'_>, source: &[u8], out: &mut Vec<String>) {
    if node.kind() == "comment" {
        return;
    }
    if node.child_count() == 0 {
        if let Ok(token) = node.utf8_text(source) {
            out.push(token.to_string());
        }
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_c_tokens(child, source, out);
    }
}

fn attach_source_definition_digests(
    source: &str,
    definitions: &BTreeMap<String, SourceDefinition>,
    digests: &BTreeMap<String, String>,
) -> String {
    let mut replacements = definitions
        .values()
        .map(|definition| {
            let digest = digests
                .get(&definition.source_name)
                .expect("each source definition has an exact digest");
            (
                definition.marker_start,
                definition.marker_end,
                render_export_begin_record(
                    definition.kind,
                    &definition.source_name,
                    &definition.symbol,
                    &definition.declaration,
                    Some(digest),
                ),
            )
        })
        .collect::<Vec<_>>();
    replacements.sort_by_key(|(start, _, _)| *start);
    let mut sealed = source.to_string();
    for (start, end, marker) in replacements.into_iter().rev() {
        sealed.replace_range(start..end, &marker);
    }
    sealed
}

fn render_sealed_header(
    program_identity: &str,
    source_digest: &str,
    declarations: &BTreeMap<String, GeneratedDeclaration>,
) -> String {
    let mut declarations = declarations.values().collect::<Vec<_>>();
    declarations.sort_by_key(|declaration| declaration.record_number);
    let payload = declarations
        .into_iter()
        .map(|declaration| {
            render_declaration_record(
                &declaration.source_name,
                &declaration.symbol,
                &declaration.declaration,
                Some(&declaration.definition_digest),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{HEADER_VERSION}\n{PROGRAM_IDENTITY_PREFIX}{}{METADATA_SUFFIX}\n{SOURCE_DIGEST_PREFIX}{source_digest}{METADATA_SUFFIX}\n{payload}",
        encode_hex(program_identity)
    )
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
    let mut declarations = parse_declarations(raw_header, ManifestMode::Raw)?;
    let raw_enveloped_source = format!(
        "{SOURCE_VERSION}\n{PROGRAM_IDENTITY_PREFIX}{}{METADATA_SUFFIX}\n{raw_source}",
        encode_hex(program_identity)
    );
    let (source_identity, definitions) =
        parse_source_manifest(&raw_enveloped_source, ManifestMode::Raw)?;
    debug_assert_eq!(source_identity, program_identity);
    validate_export_identities(program_identity, &declarations, &definitions)?;
    reject_export_macro_aliases(&raw_enveloped_source, definitions.values())?;
    let placeholder_digests = definitions
        .keys()
        .map(|source_name| (source_name.clone(), PLACEHOLDER_DIGEST.to_string()))
        .collect::<BTreeMap<_, _>>();
    let provisional_source =
        attach_source_definition_digests(&raw_enveloped_source, &definitions, &placeholder_digests);
    let (_, provisional_definitions) =
        parse_source_manifest(&provisional_source, ManifestMode::Sealed)?;
    let definition_digests =
        bind_c_definitions(&provisional_source, &provisional_definitions, false)?;
    for (source_name, declaration) in &mut declarations {
        declaration.definition_digest = definition_digests
            .get(source_name)
            .expect("every declaration has a bound source definition")
            .clone();
    }
    let sealed_source = attach_source_definition_digests(
        &provisional_source,
        &provisional_definitions,
        &definition_digests,
    );
    let digest = source_digest(&sealed_source);
    let sealed_header = render_sealed_header(program_identity, &digest, &declarations);
    GeneratedHeader::parse(&sealed_header)?.validate_source(&sealed_source)?;
    Ok((sealed_source, sealed_header))
}

pub(crate) fn reseal_generated_artifact(
    program_identity: &str,
    source: &str,
    header: &str,
) -> Result<(String, String), GeneratedHeaderError> {
    let generated = GeneratedHeader::parse(header)?;
    if generated.program_identity != program_identity {
        return Err(GeneratedHeaderError::new(format!(
            "cannot reseal program `{program_identity}` with artifact identity `{}`",
            generated.program_identity
        )));
    }
    let raw_source = strip_source_envelope(source)?;
    let sealed_source = format!(
        "{SOURCE_VERSION}\n{PROGRAM_IDENTITY_PREFIX}{}{METADATA_SUFFIX}\n{raw_source}",
        encode_hex(program_identity)
    );
    generated.validate_structure(&sealed_source)?;
    let digest = source_digest(&sealed_source);
    let sealed_header = render_sealed_header(program_identity, &digest, &generated.declarations);
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

fn parse_record_digest(digest: Option<&str>, label: &str) -> Result<String, GeneratedHeaderError> {
    let digest = digest.ok_or_else(|| {
        GeneratedHeaderError::new(format!("{label} is missing its definition digest"))
    })?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(GeneratedHeaderError::new(format!(
            "{label} definition digest is not canonical lowercase SHA-256"
        )));
    }
    Ok(digest.to_string())
}

fn source_digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

fn trim_line_ending(line: &str) -> &str {
    match line.strip_suffix('\n') {
        Some(without_newline) => match without_newline.strip_suffix('\r') {
            Some(without_carriage_return) => without_carriage_return,
            None => without_newline,
        },
        None => line,
    }
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
    definitions: impl Iterator<Item = &'a SourceDefinition>,
) -> Result<(), GeneratedHeaderError> {
    let mut protected_tokens = std::collections::BTreeSet::new();
    protected_tokens.extend(
        [
            "_Noreturn",
            "_Thread_local",
            "extern",
            "inline",
            "main",
            "static",
            "typedef",
        ]
        .into_iter()
        .map(str::to_string),
    );
    for definition in definitions {
        protected_tokens.insert(definition.symbol.clone());
        protected_tokens.extend(parse_c_declaration_tokens(&definition.declaration)?);
    }
    for logical_line in preprocessor_logical_lines(source) {
        let Some(name) = macro_definition_name(&logical_line) else {
            continue;
        };
        if protected_tokens.contains(&name) {
            return Err(GeneratedHeaderError::new(format!(
                "generated source preprocessor-rebinds exported declaration token `{name}`"
            )));
        }
        if macro_definition_replacement(&logical_line)
            .is_some_and(macro_replacement_can_change_declaration_structure)
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated source macro `{name}` can manufacture declaration structure"
            )));
        }
    }
    Ok(())
}

fn is_preprocessor_directive(line: &str) -> bool {
    preprocessor_directive_name(line).is_some()
}

fn preprocessor_directive_name(line: &str) -> Option<String> {
    let mut cursor = CDirectiveCursor::new(line);
    cursor.skip_trivia()?;
    cursor.take_directive_introducer()?;
    cursor.skip_trivia()?;
    cursor.identifier()
}

fn preprocessor_conditional_depth(source: &str) -> Result<usize, GeneratedHeaderError> {
    let mut depth = 0usize;
    for line in preprocessor_logical_lines(source) {
        match preprocessor_directive_name(&line).as_deref() {
            Some("if" | "ifdef" | "ifndef") => depth += 1,
            Some("elif" | "else") if depth == 0 => {
                return Err(GeneratedHeaderError::new(
                    "generated source has a preprocessor branch without an opening conditional",
                ));
            }
            Some("endif") => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    GeneratedHeaderError::new(
                        "generated source has a preprocessor end without an opening conditional",
                    )
                })?;
            }
            _ => {}
        }
    }
    Ok(depth)
}

fn validate_generated_include_set(source: &str) -> Result<(), GeneratedHeaderError> {
    const ALLOWED_INCLUDES: &[&str] = &[
        "\"chelis_blas.h\"",
        "\"chelis_math.h\"",
        "\"chelis_runtime.h\"",
        "<assert.h>",
        "<inttypes.h>",
        "<math.h>",
        "<pthread.h>",
        "<stdio.h>",
        "<stdlib.h>",
        "<string.h>",
    ];
    for line in preprocessor_logical_lines(source) {
        if preprocessor_directive_name(&line).as_deref() != Some("include") {
            continue;
        }
        let mut cursor = CDirectiveCursor::new(&line);
        cursor.skip_trivia();
        cursor.take_directive_introducer();
        cursor.skip_trivia();
        cursor.identifier();
        let target = cursor.remaining.trim();
        if !ALLOWED_INCLUDES.contains(&target) {
            return Err(GeneratedHeaderError::new(format!(
                "generated source includes unrecognized header `{target}`"
            )));
        }
    }
    Ok(())
}

fn validate_generated_pragmas(source: &str) -> Result<(), GeneratedHeaderError> {
    if strip_c_comments(&splice_c_lines(source)).contains("_Pragma") {
        return Err(GeneratedHeaderError::new(
            "generated source uses unsupported `_Pragma` preprocessing",
        ));
    }
    for line in preprocessor_logical_lines(source) {
        if preprocessor_directive_name(&line).as_deref() != Some("pragma") {
            continue;
        }
        let payload = preprocessor_directive_payload(&line, "pragma").unwrap_or_default();
        if !["omp ", "clang diagnostic ", "GCC diagnostic "]
            .into_iter()
            .any(|prefix| payload.starts_with(prefix))
        {
            return Err(GeneratedHeaderError::new(format!(
                "generated source uses unsupported pragma `{payload}`"
            )));
        }
    }
    Ok(())
}

fn preprocessor_directive_payload<'a>(line: &'a str, expected: &str) -> Option<&'a str> {
    let mut cursor = CDirectiveCursor::new(line);
    cursor.skip_trivia()?;
    cursor.take_directive_introducer()?;
    cursor.skip_trivia()?;
    if cursor.identifier()? != expected {
        return None;
    }
    Some(cursor.remaining.trim())
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
        let splice_marker_len = if bytes[index] == b'\\' {
            Some(1)
        } else if bytes[index..].starts_with(b"??/") {
            // C replaces trigraphs before removing escaped newlines, so
            // `??/` is a backslash for the purpose of line splicing.
            Some(3)
        } else {
            None
        };
        if let Some(marker_len) = splice_marker_len {
            if bytes.get(index + marker_len) == Some(&b'\n') {
                index += marker_len + 1;
                continue;
            }
            if bytes.get(index + marker_len) == Some(&b'\r')
                && bytes.get(index + marker_len + 1) == Some(&b'\n')
            {
                index += marker_len + 2;
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

fn macro_definition_replacement(line: &str) -> Option<&str> {
    let mut cursor = CDirectiveCursor::new(line);
    cursor.skip_trivia()?;
    cursor.take_directive_introducer()?;
    cursor.skip_trivia()?;
    if cursor.identifier()? != "define" {
        return None;
    }
    cursor.skip_trivia()?;
    cursor.identifier()?;
    Some(cursor.remaining.trim_start())
}

fn macro_replacement_can_change_declaration_structure(replacement: &str) -> bool {
    ["{", "}", "<%", "%>", "??<", "??>", "##", "%:%:", "??=??="]
        .into_iter()
        .any(|token| replacement.contains(token))
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
        render_declaration, render_declaration_record, render_direct_export_begin,
        render_direct_export_end, reseal_generated_artifact, seal_generated_artifact,
        source_digest,
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

        let beta = complete.declaration("beta").expect("beta declaration");
        let partial_header = complete_header.replace(
            &render_declaration_record(
                beta.source_name(),
                beta.symbol(),
                beta.declaration(),
                Some(beta.definition_digest()),
            ),
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
    fn generated_header_requires_a_bijection_between_exports_and_external_symbols() {
        let authored_source_name = "alpha";
        let shared_symbol = "chelis_fn_616c706861";
        let declaration = "int chelis_fn_616c706861(int x);";
        let header = format!(
            "{}\n{}",
            render_declaration(authored_source_name, shared_symbol, declaration),
            render_declaration(shared_symbol, shared_symbol, declaration)
        );
        let (_, authored) = raw_export(
            authored_source_name,
            shared_symbol,
            declaration,
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let direct = format!(
            "{}\nint chelis_fn_616c706861(int x) {{\n    return x + 2;\n}}\n{}",
            render_direct_export_begin(shared_symbol, shared_symbol, declaration),
            render_direct_export_end(shared_symbol)
        );
        assert!(
            seal_generated_artifact("demo", &format!("{authored}\n{direct}\n"), &header).is_err(),
            "distinct canonical authored/direct records must not claim one external C symbol"
        );

        let direct_symbol = "direct_entry";
        let direct_declaration = "int direct_entry(int x);";
        let distinct_header = format!(
            "{}\n{}",
            render_declaration(authored_source_name, shared_symbol, declaration),
            render_declaration(direct_symbol, direct_symbol, direct_declaration)
        );
        let distinct_direct = format!(
            "{}\nint direct_entry(int x) {{\n    return x + 2;\n}}\n{}",
            render_direct_export_begin(direct_symbol, direct_symbol, direct_declaration),
            render_direct_export_end(direct_symbol)
        );
        seal_generated_artifact(
            "demo",
            &format!("{authored}\n{distinct_direct}\n"),
            &distinct_header,
        )
        .expect("distinct canonical external symbols remain valid");
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

    fn rewrite_source_digest(header: &str, source: &str) -> String {
        let mut lines = header.lines();
        let version = lines.next().expect("header version");
        let identity = lines.next().expect("program identity");
        let _old_digest = lines.next().expect("source digest");
        format!(
            "{version}\n{identity}\n/* chelis-source-sha256: {} */\n{}",
            source_digest(source),
            lines.collect::<Vec<_>>().join("\n")
        )
    }

    fn two_export_fixture() -> (String, String) {
        let alpha_decl = "int chelis_fn_616c706861(int x);";
        let beta_decl = "int chelis_fn_62657461(int x);";
        let header = format!(
            "{}\n{}",
            render_declaration("alpha", "chelis_fn_616c706861", alpha_decl),
            render_declaration("beta", "chelis_fn_62657461", beta_decl)
        );
        let (_, alpha) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            alpha_decl,
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let (_, beta) = raw_export(
            "beta",
            "chelis_fn_62657461",
            beta_decl,
            "int chelis_fn_62657461(int x) {\n    return x + 100;\n}",
        );
        seal_generated_artifact("demo", &format!("{alpha}\n{beta}\n"), &header)
            .expect("two-export fixture")
    }

    #[test]
    fn sealed_artifact_binds_each_record_to_its_exact_definition() {
        let (source, header) = two_export_fixture();
        let alpha = "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}";
        let beta = "int chelis_fn_62657461(int x) {\n    return x + 100;\n}";
        let swapped = source
            .replacen(alpha, "__alpha_definition__", 1)
            .replacen(beta, alpha, 1)
            .replacen("__alpha_definition__", beta, 1);
        let resealed = rewrite_source_digest(&header, &swapped);
        assert!(
            GeneratedHeader::parse(&resealed)
                .expect("syntactically resealed header")
                .validate_source(&swapped)
                .is_err(),
            "whole-file resealing must not authorize export records containing each other's definitions"
        );
    }

    #[test]
    fn sealed_artifact_rejects_indirect_macro_reassociation() {
        let (source, header) = two_export_fixture();
        let swapped_body = source
            .replace(
                "int chelis_fn_616c706861(int x) {",
                "int ALPHA_IMPL(int x) {",
            )
            .replace("int chelis_fn_62657461(int x) {", "int BETA_IMPL(int x) {");
        let envelope_end = swapped_body
            .match_indices('\n')
            .nth(1)
            .map(|(index, _)| index + 1)
            .expect("source envelope");
        let swapped = format!(
            "{}#define ALPHA_IMPL chelis_fn_62657461\n#define BETA_IMPL chelis_fn_616c706861\n{}",
            &swapped_body[..envelope_end],
            &swapped_body[envelope_end..]
        );
        let resealed = rewrite_source_digest(&header, &swapped);
        assert!(
            GeneratedHeader::parse(&resealed)
                .expect("syntactically resealed header")
                .validate_source(&swapped)
                .is_err(),
            "indirect macros must not exchange public definitions"
        );
    }

    #[test]
    fn sealed_artifact_rejects_definition_symbol_and_linkage_spoofing() {
        let declaration = "int chelis_fn_616c706861(int x);";
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            declaration,
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
            "",
        );
        for spoofed in [
            source.replace(
                "int chelis_fn_616c706861(int x) {",
                "int another_symbol(int x) {",
            ),
            source.replace(
                "int chelis_fn_616c706861(int x) {",
                "static int chelis_fn_616c706861(int x) {",
            ),
        ] {
            let resealed = rewrite_source_digest(&header, &spoofed);
            assert!(
                GeneratedHeader::parse(&resealed)
                    .expect("syntactically resealed header")
                    .validate_source(&spoofed)
                    .is_err(),
                "metadata must bind the actual definition symbol and external linkage"
            );
        }

        let wrong_declaration = "int another_symbol(int x);";
        let declaration_spoofed_source =
            source.replace(&encode_hex(declaration), &encode_hex(wrong_declaration));
        let declaration_spoofed_header = rewrite_source_digest(
            &header
                .replace(&encode_hex(declaration), &encode_hex(wrong_declaration))
                .replace(declaration, wrong_declaration),
            &declaration_spoofed_source,
        );
        assert!(
            GeneratedHeader::parse(&declaration_spoofed_header)
                .expect("canonical symbol metadata with a different declaration is syntactic")
                .validate_source(&declaration_spoofed_source)
                .is_err(),
            "canonical symbol metadata must not validate when its declaration names another C symbol"
        );
    }

    #[test]
    fn sealed_artifact_rejects_resealed_unregistered_external_definition() {
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
            "",
        );
        let extra = format!("{source}\nint external_helper(int x) {{\n    return x - 1;\n}}\n");
        let resealed = rewrite_source_digest(&header, &extra);
        assert!(
            GeneratedHeader::parse(&resealed)
                .expect("syntactically resealed header")
                .validate_source(&extra)
                .is_err(),
            "an unregistered external definition must remain invalid after digest resealing"
        );
    }

    #[test]
    fn compiler_reseal_preserves_definition_commitments() {
        let (source, header) = sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
            "",
        );
        let compiler_owned_additions = format!(
            "{source}\nstatic/* private */ inline int helper(int x) {{ return x; }}\nint main(void) {{ return 0; }}\n"
        );
        let (resealed_source, resealed_header) =
            reseal_generated_artifact("demo", &compiler_owned_additions, &header)
                .expect("private helpers and the exact process main may be appended");
        GeneratedHeader::parse(&resealed_header)
            .expect("resealed header")
            .validate_source(&resealed_source)
            .expect("resealed compiler-owned additions");

        let changed_definition = source.replace("return x + 1;", "return x + 100;");
        assert!(
            reseal_generated_artifact("demo", &changed_definition, &header).is_err(),
            "ordinary resealing must not mint a new public-definition commitment"
        );
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
            .expect("structural validation accepts equivalent multiline C formatting");
    }

    #[test]
    fn sealed_artifact_distinguishes_function_linkage_from_body_local_static_storage() {
        sealed_fixture(
            "demo",
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    static int calls;\n    calls += 1;\n    return x + calls;\n}",
            "",
        );

        let (internal_header, internal_source) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "static int chelis_fn_616c706861(int x) {\n    return x;\n}",
        );
        assert!(
            seal_generated_artifact("demo", &internal_source, &internal_header).is_err(),
            "function-level static storage must still be rejected as internal linkage"
        );
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
    fn sealed_artifact_rejects_macro_rebound_export_signature_tokens() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "extern int chelis_fn_616c706861(int x);",
            "extern int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let source = format!("#define extern static\n{export}\n");
        assert!(
            seal_generated_artifact("demo", &source, &header).is_err(),
            "a source macro must not change an advertised external definition into internal linkage"
        );
    }

    #[test]
    fn sealed_artifact_allows_macros_unrelated_to_export_signatures() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + CHELIS_PRIVATE_BIAS;\n}",
        );
        let source = format!("#define CHELIS_PRIVATE_BIAS 1\n{export}\n");
        seal_generated_artifact("demo", &source, &header)
            .expect("private implementation macros must remain available");
    }

    #[test]
    fn sealed_artifact_rejects_macro_generated_external_definitions() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        for helper in [
            "#define EMIT_EXTERNAL_HELPER int external_helper(int x) { return x - 1; }\nEMIT_EXTERNAL_HELPER",
            "#define EXTERNAL_HELPER_DECL int external_helper(int x)\nEXTERNAL_HELPER_DECL { return x - 1; }",
            "#define static\nstatic int external_helper(int x) { return x - 1; }",
            "#define JOIN_INNER(a, b) a ## b\n#define JOIN(a, b) JOIN_INNER(a, b)\n#define LBRACE JOIN(<, %)\n#define RBRACE JOIN(%, >)\n#define EMIT_EXTERNAL int external_helper(int x) LBRACE return x - 1; RBRACE\nEMIT_EXTERNAL;",
            "#define JOIN_INNER(a, b) a %:%: b\n#define JOIN(a, b) JOIN_INNER(a, b)\n#define LBRACE JOIN(<, %)\n#define RBRACE JOIN(%, >)\n#define EMIT_EXTERNAL int external_helper(int x) LBRACE return x - 1; RBRACE\nEMIT_EXTERNAL;",
            "#define JOIN_INNER(a, b) a ??=??= b\n#define JOIN(a, b) JOIN_INNER(a, b)\n#define LBRACE JOIN(<, %)\n#define RBRACE JOIN(%, >)\n#define EMIT_EXTERNAL int external_helper(int x) LBRACE return x - 1; RBRACE\nEMIT_EXTERNAL;",
            "#define main injected_external\nint main(void) { return 0; }",
        ] {
            let source = format!("{helper}\n{export}\n");
            assert!(
                seal_generated_artifact("demo", &source, &header).is_err(),
                "preprocessing must not add an external definition outside the sealed export set"
            );
        }
    }

    #[test]
    fn sealed_artifact_rejects_conditionally_erased_exports_and_unknown_includes() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "int chelis_fn_616c706861(int x);",
            "int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        for source in [
            format!("#if 0\n{export}\n#endif\n"),
            format!("#include \"unsealed_external_helper.h\"\n{export}\n"),
            format!("#pragma weak external_alias = chelis_fn_616c706861\n{export}\n"),
            format!("_Pragma(\"weak external_alias = chelis_fn_616c706861\")\n{export}\n"),
        ] {
            assert!(
                seal_generated_artifact("demo", &source, &header).is_err(),
                "preprocessing must not erase a sealed export or add definitions from an unknown header"
            );
        }
    }

    #[test]
    fn sealed_artifact_rejects_trigraph_spliced_signature_macro() {
        let (header, export) = raw_export(
            "alpha",
            "chelis_fn_616c706861",
            "extern int chelis_fn_616c706861(int x);",
            "extern int chelis_fn_616c706861(int x) {\n    return x + 1;\n}",
        );
        let source = format!("#def??/\nine extern static\n{export}\n");
        assert!(
            seal_generated_artifact("demo", &source, &header).is_err(),
            "C trigraph replacement plus line splicing must not bypass signature-token protection"
        );
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
