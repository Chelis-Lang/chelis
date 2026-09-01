//! Shared fail-closed C-family source classification.
//!
//! The runtime-representation and capacity censuses use the same lexical
//! normalization, type vocabulary, and alias resolution.  Carrier discovery
//! is token- and structure-based: qualifier placement and whitespace never
//! decide whether a pointer, array, cast, or `sizeof(type)` is visible.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use clang::{
    Clang, Entity, EntityKind, Index, Type, TypeKind, Unsaved, diagnostic::Severity,
    source::SourceRange,
};
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

pub const NUMERIC_C_TYPES: &[&str] = &[
    "double",
    "float",
    "int",
    "short",
    "long",
    "signed",
    "unsigned",
    "size_t",
    "ptrdiff_t",
    "intptr_t",
    "uintptr_t",
    "int64_t",
    "int32_t",
    "int16_t",
    "int8_t",
    "uint64_t",
    "uint32_t",
    "uint16_t",
    "uint8_t",
];

pub const NON_NUMERIC_C_TYPE_WORDS: &[&str] = &[
    "_Bool",
    "_Noreturn",
    "bool",
    "char",
    "const",
    "enum",
    "extern",
    "inline",
    "register",
    "restrict",
    "static",
    "struct",
    "typedef",
    "union",
    "void",
    "volatile",
    "wchar_t",
    "__restrict",
    "__restrict__",
];

const ELEMENT_C_TYPES: &[&str] = &[
    "_Bool",
    "bool",
    "char",
    "double",
    "float",
    "half",
    "__half",
    "hip_bfloat16",
    "bfloat",
    "int",
    "short",
    "long",
    "signed",
    "unsigned",
    "int64_t",
    "int32_t",
    "int16_t",
    "int8_t",
    "uint64_t",
    "uint32_t",
    "uint16_t",
    "uint8_t",
];

const QUALIFIERS: &[&str] = &[
    "_Nonnull",
    "_Null_unspecified",
    "_Nullable",
    "const",
    "volatile",
    "restrict",
    "__restrict",
    "__restrict__",
    "__unsafe_unretained",
    "CHELIS_RESTRICT",
];

/// Exact external nonnumeric spellings accepted at carrier positions.  This
/// includes opaque handles and SDK control enums.  The vocabulary is closed:
/// adding a spelling is a reviewed classifier change, never a silent non-match.
const EXTERNAL_NONNUMERIC_C_TYPES: &[&str] = &[
    "FILE",
    "MPSMatrix",
    "MPSMatrixDescriptor",
    "MPSMatrixMultiplication",
    "MTLBuffer",
    "MTLGPUFamily",
    "NSMutableDictionary",
    "NSError",
    "NSString",
    "dispatch_once_t",
    "hipError_t",
    "hipFunction_t",
    "hipModule_t",
    "hipblasHandle_t",
    "hipblasStatus_t",
    "hiprtcProgram",
    "hiprtcResult",
    "id",
];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CarrierUse {
    pub kind: String,
    pub owner: String,
    pub signature: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanErrorKind {
    MalformedCandidate,
    UnknownType,
    IncompleteFragment,
    InvalidRust,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanError {
    pub kind: ScanErrorKind,
    pub message: String,
}

impl ScanError {
    fn new(kind: ScanErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ScanError {}

fn cxx_raw_string_end(chars: &[char], start: usize) -> Option<Result<usize, ()>> {
    let prefixes: &[&[char]] = &[
        &['R'],
        &['u', '8', 'R'],
        &['u', 'R'],
        &['U', 'R'],
        &['L', 'R'],
        &['@', 'R'],
    ];
    let prefix = prefixes.iter().find(|prefix| {
        chars.get(start..start + prefix.len()) == Some(**prefix)
            && chars.get(start + prefix.len()) == Some(&'"')
    })?;
    let quote = start + prefix.len();
    let delimiter_start = quote + 1;
    let mut open = delimiter_start;
    while open < chars.len() && chars[open] != '(' {
        if open - delimiter_start == 16
            || chars[open].is_whitespace()
            || matches!(chars[open], ')' | '\\')
        {
            return Some(Err(()));
        }
        open += 1;
    }
    if open == chars.len() {
        return Some(Err(()));
    }
    let delimiter = &chars[delimiter_start..open];
    let mut cursor = open + 1;
    while cursor < chars.len() {
        if chars[cursor] == ')'
            && chars.get(cursor + 1..cursor + 1 + delimiter.len()) == Some(delimiter)
            && chars.get(cursor + 1 + delimiter.len()) == Some(&'"')
        {
            return Some(Ok(cursor + delimiter.len() + 2));
        }
        cursor += 1;
    }
    Some(Err(()))
}

fn validate_c_lexical_closure(text: &str) -> Result<(), ScanError> {
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if let Some(raw_end) = cxx_raw_string_end(&chars, index) {
            index = raw_end.map_err(|()| {
                ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    "unterminated C++ raw string literal",
                )
            })?;
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'*') {
            index += 2;
            let mut closed = false;
            while index < chars.len() {
                if chars[index] == '*' && chars.get(index + 1) == Some(&'/') {
                    index += 2;
                    closed = true;
                    break;
                }
                index += 1;
            }
            if !closed {
                return Err(ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    "unterminated C block comment",
                ));
            }
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'/') {
            index += 2;
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }
        if chars[index] == '"' || chars[index] == '\'' {
            let quote = chars[index];
            index += 1;
            let mut closed = false;
            while index < chars.len() {
                let current = chars[index];
                index += 1;
                if current == '\\' && index < chars.len() {
                    index += 1;
                } else if current == quote {
                    closed = true;
                    break;
                } else if current == '\n' {
                    break;
                }
            }
            if !closed {
                return Err(ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    "unterminated C string or character literal",
                ));
            }
            continue;
        }
        index += 1;
    }
    Ok(())
}

/// Remove C/C++ comments without interpreting comment markers inside string
/// or character literals.  A block comment becomes one space and newlines are
/// retained, which keeps adjacent tokens separate and diagnostics attributable.
pub fn strip_c_comments(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        if let Some(Ok(raw_end)) = cxx_raw_string_end(&chars, index) {
            output.extend(chars[index..raw_end].iter());
            index = raw_end;
            continue;
        }
        if chars[index] == '"' || chars[index] == '\'' {
            let quote = chars[index];
            output.push(quote);
            index += 1;
            while index < chars.len() {
                let current = chars[index];
                output.push(current);
                index += 1;
                if current == '\\' && index < chars.len() {
                    output.push(chars[index]);
                    index += 1;
                } else if current == quote {
                    break;
                }
            }
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'*') {
            output.push(' ');
            index += 2;
            let mut closed = false;
            while index < chars.len() {
                if chars[index] == '*' && chars.get(index + 1) == Some(&'/') {
                    index += 2;
                    closed = true;
                    break;
                }
                if chars[index] == '\n' {
                    output.push('\n');
                }
                index += 1;
            }
            if !closed {
                break;
            }
            continue;
        }
        if chars[index] == '/' && chars.get(index + 1) == Some(&'/') {
            index += 2;
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }
        output.push(chars[index]);
        index += 1;
    }
    output
}

/// Tokenize the C-family subset used by the repository.  Identities are token
/// sequences, not whitespace or preprocessor pretty-printing.
pub fn canonical_c_tokens(source: &str) -> String {
    lex_c_tokens(source).join(" ")
}

fn lex_c_tokens(source: &str) -> Vec<String> {
    let source = strip_c_comments(source);
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let current = chars[index];
        if current.is_whitespace() {
            index += 1;
            continue;
        }
        if let Some(Ok(raw_end)) = cxx_raw_string_end(&chars, index) {
            tokens.push(chars[index..raw_end].iter().collect());
            index = raw_end;
            continue;
        }
        if current == '"' || current == '\'' {
            let quote = current;
            let mut token = String::from(current);
            index += 1;
            while index < chars.len() {
                let next = chars[index];
                token.push(next);
                index += 1;
                if next == '\\' && index < chars.len() {
                    token.push(chars[index]);
                    index += 1;
                } else if next == quote {
                    break;
                }
            }
            tokens.push(token);
            continue;
        }
        if current.is_ascii_alphabetic() || current == '_' {
            let mut token = String::new();
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
            {
                token.push(chars[index]);
                index += 1;
            }
            tokens.push(if token == "bool" {
                "_Bool".to_string()
            } else {
                token
            });
            continue;
        }
        if index + 2 < chars.len() && chars[index..index + 3] == ['.', '.', '.'] {
            tokens.push("...".to_string());
            index += 3;
            continue;
        }
        if current.is_ascii_digit()
            || (current == '.' && chars.get(index + 1).is_some_and(char::is_ascii_digit))
        {
            let mut token = String::new();
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric()
                    || chars[index] == '_'
                    || chars[index] == '.')
            {
                token.push(chars[index]);
                index += 1;
            }
            tokens.push(token);
            continue;
        }
        if index + 1 < chars.len() {
            let pair = [current, chars[index + 1]].iter().collect::<String>();
            if matches!(
                pair.as_str(),
                "->" | "++"
                    | "--"
                    | "<<"
                    | ">>"
                    | "<="
                    | ">="
                    | "=="
                    | "!="
                    | "&&"
                    | "||"
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "%="
                    | "&="
                    | "|="
                    | "^="
            ) {
                tokens.push(pair);
                index += 2;
                continue;
            }
        }
        tokens.push(current.to_string());
        index += 1;
    }
    tokens
}

fn tokens(source: &str) -> Vec<String> {
    lex_c_tokens(source)
}

const ALIAS_DEFINITION_PREFIX: &str = "__chelis_alias_definition_";

fn encode_alias_definition(definition: &str) -> String {
    let mut encoded = String::from(ALIAS_DEFINITION_PREFIX);
    for byte in definition.as_bytes() {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

fn decode_alias_definition(word: &str) -> Option<String> {
    let encoded = word.strip_prefix(ALIAS_DEFINITION_PREFIX)?;
    if encoded.len() % 2 != 0 {
        return None;
    }
    let bytes = (0..encoded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&encoded[index..index + 2], 16).ok())
        .collect::<Option<Vec<_>>>()?;
    String::from_utf8(bytes).ok()
}

pub fn collect_typedefs(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut aliases = BTreeMap::new();
    collect_aggregate_typedefs(text, &mut aliases);
    for statement in strip_c_comments(text).split(';') {
        let canonical = canonical_c_tokens(statement);
        let statement_tokens: Vec<&str> = canonical.split_whitespace().collect();
        let Some(typedef_index) = statement_tokens
            .iter()
            .rposition(|token| *token == "typedef")
        else {
            continue;
        };
        let statement = statement_tokens[typedef_index..].join(" ");
        let Some(rest) = statement.strip_prefix("typedef ") else {
            continue;
        };
        if rest.contains('{') {
            continue;
        }
        if rest.contains('(') {
            if let Some((name, mut target)) = function_pointer_typedef(&statement) {
                target.push(encode_alias_definition(&format!("{statement};")));
                merge_alias_words(&mut aliases, name, target);
            }
            continue;
        }
        if rest.contains('[') {
            if let Some((name, mut target)) = array_typedef(&statement) {
                target.push(encode_alias_definition(&format!("{statement};")));
                merge_alias_words(&mut aliases, name, target);
            }
            continue;
        }
        let mut words: Vec<String> = rest
            .split(|character: char| !(character.is_alphanumeric() || character == '_'))
            .filter(|word| !word.is_empty())
            .map(str::to_string)
            .collect();
        if words.len() >= 2 {
            let name = words.pop().expect("length checked");
            if words
                .first()
                .is_some_and(|word| matches!(word.as_str(), "struct" | "enum" | "union"))
            {
                words.truncate(1);
            }
            words.push(encode_alias_definition(&format!("{statement};")));
            merge_alias_words(&mut aliases, name, words);
        }
    }
    aliases
}

fn collect_aggregate_typedefs(text: &str, aliases: &mut BTreeMap<String, Vec<String>>) {
    let source_tokens = tokens(text);
    let mut index = 0usize;
    while index + 2 < source_tokens.len() {
        if source_tokens[index] != "typedef"
            || !matches!(
                source_tokens[index + 1].as_str(),
                "struct" | "enum" | "union"
            )
        {
            index += 1;
            continue;
        }
        let aggregate = source_tokens[index + 1].clone();
        let Some(open_offset) = source_tokens[index + 2..]
            .iter()
            .position(|token| token == "{" || token == ";")
        else {
            break;
        };
        let open = index + 2 + open_offset;
        if source_tokens[open] != "{" {
            index = open + 1;
            continue;
        }
        let mut depth = 1usize;
        let mut close = open + 1;
        while close < source_tokens.len() && depth > 0 {
            match source_tokens[close].as_str() {
                "{" => depth += 1,
                "}" => depth -= 1,
                _ => {}
            }
            close += 1;
        }
        if depth != 0 {
            break;
        }
        let Some(semicolon_offset) = source_tokens[close..].iter().position(|token| token == ";")
        else {
            break;
        };
        let semicolon = close + semicolon_offset;
        if let Some(name) = source_tokens[close..semicolon]
            .iter()
            .rev()
            .find(|token| is_identifier(token))
        {
            merge_alias_words(aliases, name.clone(), vec![aggregate]);
        }
        index = semicolon + 1;
    }
}

fn merge_alias_words(
    aliases: &mut BTreeMap<String, Vec<String>>,
    name: String,
    target: Vec<String>,
) {
    let resolved = aliases.entry(name).or_default();
    for word in target {
        if !resolved.contains(&word) {
            resolved.push(word);
        }
    }
}

fn array_typedef(statement: &str) -> Option<(String, Vec<String>)> {
    let tokens = tokens(statement);
    let open = tokens.iter().position(|token| token == "[")?;
    let name = tokens.get(open.checked_sub(1)?)?;
    if !is_identifier(name)
        || NUMERIC_C_TYPES.contains(&name.as_str())
        || NON_NUMERIC_C_TYPE_WORDS.contains(&name.as_str())
    {
        return None;
    }
    let target: Vec<String> = tokens[..open - 1]
        .iter()
        .filter(|token| token.as_str() != "typedef" && is_identifier(token))
        .cloned()
        .collect();
    (!target.is_empty()).then(|| (name.clone(), target))
}

fn function_pointer_typedef(statement: &str) -> Option<(String, Vec<String>)> {
    let tokens = tokens(statement);
    let anchor = tokens.windows(4).position(|window| {
        window[0] == "(" && window[1] == "*" && is_identifier(&window[2]) && window[3] == ")"
    })?;
    let name = tokens[anchor + 2].clone();
    let target = tokens
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != anchor + 2)
        .map(|(_, token)| token.clone())
        .filter(|token| token != "typedef" && is_identifier(token))
        .collect();
    Some((name, target))
}

pub fn resolve_words(words: Vec<String>, aliases: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut output = words;
    for _ in 0..8 {
        let mut changed = false;
        let mut next = Vec::with_capacity(output.len());
        for word in &output {
            if let Some(target) = aliases.get(word) {
                next.extend(
                    target
                        .iter()
                        .filter(|word| decode_alias_definition(word).is_none())
                        .cloned(),
                );
                changed = true;
            } else {
                next.push(word.clone());
            }
        }
        output = next;
        if !changed {
            break;
        }
    }
    output
}

pub fn classify(signature: &str, aliases: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let words: Vec<String> = signature
        .split(|character: char| !(character.is_alphanumeric() || character == '_'))
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect();
    let words = resolve_words(words, aliases);
    let mut flags = Vec::new();
    if words.iter().any(|word| word == "double" || word == "float") {
        flags.push("float-carrier".to_string());
    }
    if words
        .windows(2)
        .any(|pair| pair[0] == "int" && pair[1].contains("dtype"))
    {
        flags.push("raw-dtype-int".to_string());
    }
    if words
        .iter()
        .any(|word| NUMERIC_C_TYPES.contains(&word.as_str()))
    {
        flags.push("numeric-op".to_string());
    }
    flags
}

fn is_identifier(token: &str) -> bool {
    let mut characters = token.chars();
    characters
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn collect_macro_aliases(source: &str) -> BTreeMap<String, Vec<String>> {
    let mut aliases = BTreeMap::new();
    for line in strip_c_comments(source).lines() {
        let Some(rest) = line.trim_start().strip_prefix("#define") else {
            continue;
        };
        let rest = rest.trim_start();
        let name_end = rest
            .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .unwrap_or(rest.len());
        if name_end == 0 {
            continue;
        }
        let name = &rest[..name_end];
        let mut replacement = &rest[name_end..];
        if replacement.starts_with('(') {
            let Some(close) = replacement.find(')') else {
                continue;
            };
            replacement = &replacement[close + 1..];
        }
        let mut words: Vec<String> = tokens(replacement)
            .into_iter()
            .filter(|token| is_identifier(token))
            .collect();
        if !words.is_empty() {
            words.push(encode_alias_definition(line.trim()));
            merge_alias_words(&mut aliases, name.to_string(), words);
        }
    }
    aliases
}

/// Collect every typedef and type-like macro definition without allowing a
/// later definition to erase an earlier meaning.  Repository scans use the
/// union from all tracked C-family sources as a conservative include prelude.
pub fn collect_c_aliases(source: &str) -> BTreeMap<String, Vec<String>> {
    let mut aliases = collect_typedefs(source);
    for (name, target) in collect_macro_aliases(source) {
        merge_alias_words(&mut aliases, name, target);
    }
    aliases
}

#[derive(Clone, Debug)]
struct TypeSpec {
    start: usize,
    authored: String,
    resolved: String,
    qualifiers: BTreeSet<String>,
}

fn resolve_element_type(
    authored: &str,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<Option<String>, ScanError> {
    resolve_type_words(
        vec![authored.to_string()],
        aliases,
        is_unknown_type_word(authored),
    )
}

fn resolve_type_words(
    authored: Vec<String>,
    aliases: &BTreeMap<String, Vec<String>>,
    reject_unknown: bool,
) -> Result<Option<String>, ScanError> {
    let words = resolve_words_checked(authored, aliases)?;
    if reject_unknown
        && let Some(unknown) = words.iter().find(|word| {
            !ELEMENT_C_TYPES.contains(&word.as_str())
                && !NON_NUMERIC_C_TYPE_WORDS.contains(&word.as_str())
                && !EXTERNAL_NONNUMERIC_C_TYPES.contains(&word.as_str())
                && !QUALIFIERS.contains(&word.as_str())
                && !is_base_modifier(word)
        })
    {
        return Err(ScanError::new(
            ScanErrorKind::UnknownType,
            format!("unknown C type word `{unknown}` at a carrier position"),
        ));
    }
    if words
        .iter()
        .any(|word| ELEMENT_C_TYPES.contains(&word.as_str()))
    {
        return Ok(Some(words.join(" ")));
    }
    if words.iter().all(|word| {
        EXTERNAL_NONNUMERIC_C_TYPES.contains(&word.as_str())
            || NON_NUMERIC_C_TYPE_WORDS.contains(&word.as_str())
    }) {
        return Ok(None);
    }
    if reject_unknown && words.iter().any(|word| is_unknown_type_word(word)) {
        return Err(ScanError::new(
            ScanErrorKind::UnknownType,
            format!(
                "unknown C type word `{}` at a carrier position",
                words
                    .iter()
                    .find(|word| is_unknown_type_word(word))
                    .expect("unknown word exists")
            ),
        ));
    }
    Ok(None)
}

fn resolves_to_known_nonnumeric(
    authored: &str,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<bool, ScanError> {
    let words = resolve_words_checked(vec![authored.to_string()], aliases)?;
    Ok(words.iter().all(|word| {
        EXTERNAL_NONNUMERIC_C_TYPES.contains(&word.as_str())
            || NON_NUMERIC_C_TYPE_WORDS.contains(&word.as_str())
    }))
}

fn range_resolves_to_known_nonnumeric(
    source_tokens: &[String],
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<bool, ScanError> {
    let identifiers: Vec<&String> = source_tokens
        .iter()
        .filter(|token| is_identifier(token))
        .collect();
    if identifiers.is_empty() {
        return Ok(false);
    }
    for identifier in identifiers {
        if !resolves_to_known_nonnumeric(identifier, aliases)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn resolve_words_checked(
    words: Vec<String>,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, ScanError> {
    let mut output = words;
    for _ in 0..32 {
        let mut changed = false;
        let mut next = Vec::with_capacity(output.len());
        for word in &output {
            if let Some(target) = aliases.get(word) {
                next.extend(
                    target
                        .iter()
                        .filter(|word| decode_alias_definition(word).is_none())
                        .cloned(),
                );
                changed = true;
            } else {
                next.push(word.clone());
            }
        }
        output = next;
        if !changed {
            return Ok(output);
        }
    }
    Err(ScanError::new(
        ScanErrorKind::UnknownType,
        "C type alias chain is cyclic or exceeds 32 expansions",
    ))
}

fn is_unknown_type_word(word: &str) -> bool {
    let lowercase = word.to_ascii_lowercase();
    word.starts_with("_Float")
        || word.starts_with("_Decimal")
        || word.starts_with("__fp")
        || word.starts_with("__bf")
        || word.starts_with("__int")
        || word.ends_with("_t")
        || [
            "bfloat", "decimal", "double", "dtype", "float", "half", "integer", "numeric", "real",
            "scalar",
        ]
        .iter()
        .any(|marker| lowercase.contains(marker))
}

fn format_hole_before(tokens: &[String], end: usize) -> Option<(usize, String)> {
    if tokens.get(end)? != "}" {
        return None;
    }
    let mut index = end;
    while index > 0 {
        index -= 1;
        if tokens[index] == "{" {
            let contents = &tokens[index + 1..end];
            let name = contents
                .first()
                .filter(|name| is_identifier(name))
                .map_or("positional", String::as_str);
            return (contents.is_empty() || is_identifier(name))
                .then(|| (index, format!("format-hole:{name}")));
        }
        if tokens[index] == "}" {
            return None;
        }
    }
    None
}

fn format_hole_after(tokens: &[String], start: usize) -> Option<(usize, String)> {
    if tokens.get(start)? != "{" {
        return None;
    }
    let close = tokens[start + 1..].iter().position(|token| token == "}")? + start + 1;
    let contents = &tokens[start + 1..close];
    let name = contents
        .first()
        .filter(|name| is_identifier(name))
        .map_or("positional", String::as_str);
    if !contents.is_empty() && !is_identifier(name) {
        return None;
    }
    Some((close + 1, format!("format-hole:{name}")))
}

fn matching_open_reverse(
    tokens: &[String],
    close: usize,
    open_token: &str,
    close_token: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    for index in (0..=close).rev() {
        if tokens[index] == close_token {
            depth += 1;
        } else if tokens[index] == open_token {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn matching_angle_open_reverse(tokens: &[String], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    for index in (0..=close).rev() {
        match tokens[index].as_str() {
            ">" => depth += 1,
            ">>" => depth += 2,
            "<" => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            "<<" => {
                depth = depth.checked_sub(2)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn is_base_modifier(token: &str) -> bool {
    matches!(
        token,
        "_Complex" | "_Imaginary" | "signed" | "unsigned" | "short" | "long"
    )
}

fn is_storage_specifier(token: &str) -> bool {
    matches!(
        token,
        "extern" | "inline" | "register" | "static" | "typedef"
    )
}

fn is_declaration_prefix_stop(token: &str) -> bool {
    matches!(
        token,
        "break"
            | "case"
            | "continue"
            | "define"
            | "elif"
            | "else"
            | "endif"
            | "for"
            | "if"
            | "ifdef"
            | "ifndef"
            | "include"
            | "pragma"
            | "return"
            | "sizeof"
            | "switch"
            | "undef"
            | "while"
    )
}

fn type_start_before(tokens: &[String], end: usize) -> usize {
    if end == 0 {
        return 0;
    }
    let mut cursor = end - 1;
    if matches!(tokens[cursor].as_str(), ">" | ">>")
        && let Some(open) = matching_angle_open_reverse(tokens, cursor)
        && open > 0
    {
        cursor = open - 1;
    } else if tokens[cursor] == ")"
        && let Some(open) = matching_open_reverse(tokens, cursor, "(", ")")
        && open > 0
        && is_identifier(&tokens[open - 1])
    {
        cursor = open - 1;
    }
    while is_identifier(&tokens[cursor])
        && cursor > 0
        && is_identifier(&tokens[cursor - 1])
        && !is_declaration_prefix_stop(&tokens[cursor - 1])
        && !is_storage_specifier(&tokens[cursor - 1])
    {
        cursor -= 1;
    }
    cursor
}

fn declaration_context(tokens: &[String], type_end: usize) -> bool {
    let start = type_start_before(tokens, type_end);
    if tokens.get(start).is_some_and(|token| {
        matches!(
            token.as_str(),
            "break" | "case" | "continue" | "else" | "for" | "if" | "return" | "switch" | "while"
        )
    }) {
        return false;
    }
    if start == 0 {
        return true;
    }
    let previous = tokens[start - 1].as_str();
    if matches!(previous, ";" | "{" | "}" | ",") {
        return true;
    }
    if previous != "(" {
        return matches!(
            previous,
            "extern" | "inline" | "register" | "static" | "typedef"
        );
    }

    if start >= 2
        && matches!(
            tokens[start - 2].as_str(),
            "for" | "if" | "return" | "sizeof" | "switch" | "while"
        )
    {
        return false;
    }
    if start >= 3
        && tokens[start - 2] == "("
        && matches!(
            tokens[start - 3].as_str(),
            "for" | "if" | "return" | "sizeof" | "switch" | "while"
        )
    {
        return false;
    }

    // A parenthesis after a function name is a parameter-list boundary only
    // when that name itself has a declaration prefix.  This distinguishes
    // `f(type *p)` from an expression such as `f(lhs * rhs)`.
    if start >= 2 && is_identifier(&tokens[start - 2]) {
        if start < 3 {
            return false;
        }
        let before_name = tokens[start - 3].as_str();
        return ELEMENT_C_TYPES.contains(&before_name)
            || NON_NUMERIC_C_TYPE_WORDS.contains(&before_name)
            || EXTERNAL_NONNUMERIC_C_TYPES.contains(&before_name)
            || before_name == ")"
            || before_name == "*";
    }
    true
}

fn reference_declaration_context(tokens: &[String], type_end: usize) -> bool {
    let start = type_start_before(tokens, type_end);
    if start > 0 && tokens[start - 1] == "(" && (start < 2 || !is_identifier(&tokens[start - 2])) {
        return false;
    }
    declaration_context(tokens, type_end)
}

fn type_before(
    tokens: &[String],
    end: usize,
    aliases: &BTreeMap<String, Vec<String>>,
    reject_unknown_external: bool,
) -> Result<Option<TypeSpec>, ScanError> {
    if end == 0 {
        return Ok(None);
    }
    let mut cursor = end - 1;
    let mut qualifiers = BTreeSet::new();
    while QUALIFIERS.contains(&tokens[cursor].as_str()) {
        qualifiers.insert(tokens[cursor].clone());
        if cursor == 0 {
            return Ok(None);
        }
        cursor -= 1;
    }
    if tokens[cursor] == "]"
        && let Some(open) = matching_open_reverse(tokens, cursor, "[", "]")
        && tokens.get(open + 1).map(String::as_str) == Some("[")
        && tokens.get(cursor.wrapping_sub(1)).map(String::as_str) == Some("]")
    {
        return type_before(tokens, open, aliases, reject_unknown_external);
    }
    if tokens[cursor] == ")"
        && let Some(open) = matching_open_reverse(tokens, cursor, "(", ")")
        && open > 0
    {
        let marker = tokens[open - 1].as_str();
        if matches!(
            marker,
            "__attribute__" | "__declspec" | "_Alignas" | "alignas"
        ) {
            return type_before(tokens, open - 1, aliases, reject_unknown_external);
        }
        if marker == "_Atomic" {
            let mut type_spec =
                type_in_range(tokens, open + 1, cursor, aliases, reject_unknown_external)?;
            if let Some(type_spec) = &mut type_spec {
                type_spec.start = open - 1;
                while type_spec.start > 0
                    && QUALIFIERS.contains(&tokens[type_spec.start - 1].as_str())
                {
                    type_spec.start -= 1;
                    type_spec.qualifiers.insert(tokens[type_spec.start].clone());
                }
            }
            return Ok(type_spec);
        }
        if is_identifier(marker) {
            if let Some(resolved) = resolve_element_type(marker, aliases)? {
                return Ok(Some(TypeSpec {
                    start: open - 1,
                    authored: format!("{marker}()"),
                    resolved,
                    qualifiers,
                }));
            }
            if reject_unknown_external && !resolves_to_known_nonnumeric(marker, aliases)? {
                return Err(ScanError::new(
                    ScanErrorKind::UnknownType,
                    format!("unknown C type macro `{marker}(...)` at a carrier position"),
                ));
            }
            return Ok(None);
        }
    }
    if matches!(tokens[cursor].as_str(), ">" | ">>")
        && let Some(open) = matching_angle_open_reverse(tokens, cursor)
        && open > 0
        && is_identifier(&tokens[open - 1])
    {
        let container = open - 1;
        let inner = type_in_range(tokens, open + 1, cursor, aliases, false)?;
        if inner.is_some() {
            let authored = tokens[container..=cursor].join(" ");
            return Ok(Some(TypeSpec {
                start: container,
                resolved: authored.clone(),
                authored,
                qualifiers,
            }));
        }
        if resolves_to_known_nonnumeric(&tokens[container], aliases)?
            && range_resolves_to_known_nonnumeric(&tokens[open + 1..cursor], aliases)?
        {
            return Ok(None);
        }
        if reject_unknown_external {
            return Err(ScanError::new(
                ScanErrorKind::UnknownType,
                format!(
                    "unknown C++ template type `{}` at a carrier position",
                    tokens[container..=cursor].join(" ")
                ),
            ));
        }
        return Ok(None);
    }
    if let Some((start, authored)) = format_hole_before(tokens, cursor) {
        return Ok(Some(TypeSpec {
            start,
            authored: authored.clone(),
            resolved: authored,
            qualifiers,
        }));
    }
    let authored = tokens[cursor].clone();
    if EXTERNAL_NONNUMERIC_C_TYPES.contains(&authored.as_str())
        || authored == "void"
        || (cursor > 0 && matches!(tokens[cursor - 1].as_str(), "struct" | "enum" | "union"))
    {
        return Ok(None);
    }
    if !is_identifier(&authored) {
        return Ok(None);
    }
    let mut start = cursor;
    while start > 0
        && (QUALIFIERS.contains(&tokens[start - 1].as_str())
            || is_base_modifier(&tokens[start - 1]))
    {
        start -= 1;
    }
    if reject_unknown_external {
        while start > 0
            && is_identifier(&tokens[start - 1])
            && !is_declaration_prefix_stop(&tokens[start - 1])
            && !is_storage_specifier(&tokens[start - 1])
        {
            start -= 1;
        }
    }
    let authored_words: Vec<String> = tokens[start..=cursor]
        .iter()
        .filter(|word| !QUALIFIERS.contains(&word.as_str()) && !is_storage_specifier(word.as_str()))
        .cloned()
        .collect();
    if let Some(resolved) =
        resolve_type_words(authored_words.clone(), aliases, reject_unknown_external)?
    {
        for word in &tokens[start..=cursor] {
            if QUALIFIERS.contains(&word.as_str()) {
                qualifiers.insert(word.clone());
            }
        }
        return Ok(Some(TypeSpec {
            start,
            authored: authored_words.join(" "),
            resolved,
            qualifiers,
        }));
    }
    if aliases.contains_key(&authored) && resolves_to_known_nonnumeric(&authored, aliases)? {
        return Ok(None);
    }
    if cursor > 0 && tokens[cursor - 1] == "}" {
        return Ok(None);
    }
    if cursor > 0
        && is_identifier(&authored)
        && resolve_element_type(&tokens[cursor - 1], aliases)?.is_some()
    {
        return Err(ScanError::new(
            ScanErrorKind::UnknownType,
            format!("unsupported C type decoration `{authored}` at a carrier position"),
        ));
    }
    if reject_unknown_external && (is_unknown_type_word(&authored) || is_identifier(&authored)) {
        return Err(ScanError::new(
            ScanErrorKind::UnknownType,
            format!("unknown C type word `{authored}` at a carrier position"),
        ));
    }
    Ok(None)
}

fn matching_close(
    tokens: &[String],
    open: usize,
    open_token: &str,
    close_token: &str,
) -> Result<usize, ScanError> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if token == open_token {
            depth += 1;
        } else if token == close_token {
            depth = depth.checked_sub(1).ok_or_else(|| {
                ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    format!("unbalanced `{close_token}` in C carrier candidate"),
                )
            })?;
            if depth == 0 {
                return Ok(index);
            }
        }
    }
    Err(ScanError::new(
        ScanErrorKind::MalformedCandidate,
        format!("unterminated `{open_token}` in C carrier candidate"),
    ))
}

fn type_in_range(
    tokens: &[String],
    start: usize,
    end: usize,
    aliases: &BTreeMap<String, Vec<String>>,
    reject_unknown_external: bool,
) -> Result<Option<TypeSpec>, ScanError> {
    let mut qualifiers = BTreeSet::new();
    let mut index = start;
    while index < end {
        let token = &tokens[index];
        if index > start && matches!(tokens[index - 1].as_str(), "struct" | "enum" | "union") {
            index += 1;
            continue;
        }
        if token == "{"
            && index + 2 < end
            && is_identifier(&tokens[index + 1])
            && tokens[index + 2] == "}"
        {
            let authored = format!("format-hole:{}", tokens[index + 1]);
            return Ok(Some(TypeSpec {
                start,
                authored: authored.clone(),
                resolved: authored,
                qualifiers,
            }));
        }
        if QUALIFIERS.contains(&token.as_str()) {
            qualifiers.insert(token.clone());
        } else if is_identifier(token) {
            if let Some(resolved) = resolve_element_type(token, aliases)? {
                return Ok(Some(TypeSpec {
                    start,
                    authored: token.clone(),
                    resolved,
                    qualifiers,
                }));
            }
            if reject_unknown_external
                && token.chars().next().is_some_and(char::is_uppercase)
                && !EXTERNAL_NONNUMERIC_C_TYPES.contains(&token.as_str())
            {
                return Err(ScanError::new(
                    ScanErrorKind::UnknownType,
                    format!("unknown C type word `{token}` at a carrier position"),
                ));
            }
        }
        index += 1;
    }
    Ok(None)
}

fn incomplete_pointer_fragment(tokens: &[String], star: usize) -> bool {
    let mut cursor = star;
    while cursor > 0 && QUALIFIERS.contains(&tokens[cursor - 1].as_str()) {
        cursor -= 1;
    }
    cursor == 0
        && (tokens
            .get(star + 1)
            .is_some_and(|token| is_identifier(token))
            || format_hole_after(tokens, star + 1).is_some())
}

fn incomplete_array_fragment(tokens: &[String], open: usize, close: usize) -> bool {
    open > 0
        && tokens
            .get(open - 1)
            .is_some_and(|token| is_identifier(token))
        && tokens[open + 1..close].is_empty()
        && tokens[..open - 1]
            .iter()
            .all(|token| QUALIFIERS.contains(&token.as_str()))
}

fn validate_balanced(tokens: &[String]) -> Result<(), ScanError> {
    let mut stack: Vec<&str> = Vec::new();
    for token in tokens {
        match token.as_str() {
            "(" | "[" | "{" => stack.push(token),
            ")" | "]" | "}" => {
                let expected = match token.as_str() {
                    ")" => "(",
                    "]" => "[",
                    "}" => "{",
                    _ => unreachable!("closing delimiter matched above"),
                };
                if stack.pop() != Some(expected) {
                    return Err(ScanError::new(
                        ScanErrorKind::MalformedCandidate,
                        format!("unbalanced `{token}` in C carrier candidate"),
                    ));
                }
            }
            _ => {}
        }
    }
    if stack.is_empty() {
        Ok(())
    } else {
        Err(ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "unterminated delimiter in C carrier candidate",
        ))
    }
}

fn canonical_signature(
    shape: &str,
    type_spec: &TypeSpec,
    pointer_qualifiers: &BTreeSet<String>,
    name: &str,
    declarator_detail: &str,
) -> String {
    format!(
        "shape={shape};resolved={};authored={};qualifiers={};pointer-qualifiers={};name={name};declarator={declarator_detail}",
        type_spec.resolved,
        type_spec.authored,
        type_spec
            .qualifiers
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(","),
        pointer_qualifiers
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(","),
    )
}

fn has_candidate_anchor(tokens: &[String]) -> bool {
    tokens.iter().any(|token| {
        token == "*" || token == "&" || token == "&&" || token == "[" || token == "sizeof"
    })
}

fn scan_source_tokens(
    source: &str,
    owner: &str,
    require_global_balance: bool,
    predeclared_aliases: Option<&BTreeMap<String, Vec<String>>>,
) -> Result<Vec<CarrierUse>, ScanError> {
    let tokens = tokens(source);
    if !has_candidate_anchor(&tokens) {
        return Ok(Vec::new());
    }
    validate_c_lexical_closure(source)?;
    if require_global_balance {
        validate_balanced(&tokens)?;
    }
    let mut aliases = predeclared_aliases.cloned().unwrap_or_default();
    for (name, target) in collect_c_aliases(source) {
        merge_alias_words(&mut aliases, name, target);
    }
    let mut rows = Vec::new();

    for (index, token) in tokens.iter().enumerate() {
        if token == "*" || token == "&" {
            if tokens.get(index + 1).map(String::as_str) == Some("sizeof") {
                continue;
            }
            let mut type_end = index;
            let mut shape = if token == "&" { "reference" } else { "pointer" };
            if token == "*" && index > 0 && tokens[index - 1] == "(" {
                type_end = index - 1;
                shape = "function-pointer";
            }
            let reject_unknown = require_global_balance
                && if token == "&" {
                    reference_declaration_context(&tokens, type_end)
                } else {
                    declaration_context(&tokens, type_end)
                };
            let Some(type_spec) = type_before(&tokens, type_end, &aliases, reject_unknown)? else {
                if token == "*"
                    && !require_global_balance
                    && incomplete_pointer_fragment(&tokens, index)
                {
                    return Err(ScanError::new(
                        ScanErrorKind::IncompleteFragment,
                        "C carrier fragment contains a pointer declarator without its type",
                    ));
                }
                continue;
            };
            let mut cursor = index + 1;
            let mut pointer_qualifiers = BTreeSet::new();
            while cursor < tokens.len() && QUALIFIERS.contains(&tokens[cursor].as_str()) {
                pointer_qualifiers.insert(tokens[cursor].clone());
                cursor += 1;
            }
            let dynamic_name = format_hole_after(&tokens, cursor);
            let name = if let Some((_, ref name)) = dynamic_name {
                name.as_str()
            } else if tokens.get(cursor).is_some_and(|next| is_identifier(next)) {
                tokens[cursor].as_str()
            } else if tokens.get(cursor).map(String::as_str) == Some(")") {
                shape = "cast-pointer";
                "<abstract>"
            } else {
                return Err(ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    format!("parsed element pointer has no declarator after token {index}"),
                ));
            };
            rows.push(CarrierUse {
                kind: "raw-element-pointer".to_string(),
                owner: owner.to_string(),
                signature: canonical_signature(shape, &type_spec, &pointer_qualifiers, name, token),
            });
        }

        if token == "[" && index > 0 && is_identifier(&tokens[index - 1]) {
            let close = matching_close(&tokens, index, "[", "]")?;
            let name = &tokens[index - 1];
            let reject_unknown = require_global_balance && declaration_context(&tokens, index - 1);
            let Some(mut type_spec) = type_before(&tokens, index - 1, &aliases, reject_unknown)?
            else {
                if !require_global_balance && incomplete_array_fragment(&tokens, index, close) {
                    return Err(ScanError::new(
                        ScanErrorKind::IncompleteFragment,
                        "C carrier fragment contains an array declarator without its type",
                    ));
                }
                continue;
            };
            if type_spec.start > 0 && tokens[type_spec.start - 1] == "typedef" {
                type_spec.authored = name.clone();
            }
            rows.push(CarrierUse {
                kind: "raw-element-pointer".to_string(),
                owner: owner.to_string(),
                signature: canonical_signature(
                    "array",
                    &type_spec,
                    &BTreeSet::new(),
                    name,
                    &tokens[index + 1..close].join(" "),
                ),
            });
        }

        if token == "sizeof" && tokens.get(index + 1).map(String::as_str) == Some("(") {
            let close = matching_close(&tokens, index + 1, "(", ")")?;
            let Some(type_spec) =
                type_in_range(&tokens, index + 2, close, &aliases, require_global_balance)?
            else {
                continue;
            };
            rows.push(CarrierUse {
                kind: "width-arithmetic".to_string(),
                owner: owner.to_string(),
                signature: canonical_signature(
                    "sizeof",
                    &type_spec,
                    &BTreeSet::new(),
                    "<abstract>",
                    "sizeof",
                ),
            });
        }
    }
    Ok(rows)
}

static CLANG_SCAN_LOCK: Mutex<()> = Mutex::new(());

const CLANG_PRELUDE: &str = r#"
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef struct __chelis_surface_FILE FILE;
typedef unsigned char chelis_dtype;
typedef unsigned long MTLGPUFamily;
enum {
    MTLGPUFamilyApple1 = 1,
    MTLGPUFamilyApple2 = 2,
    MTLGPUFamilyApple3 = 3,
    MTLGPUFamilyApple4 = 4,
    MTLGPUFamilyApple5 = 5,
    MTLGPUFamilyApple6 = 6,
    MTLGPUFamilyApple7 = 7,
    MTLGPUFamilyApple8 = 8,
    MTLGPUFamilyApple9 = 9
};
typedef long dispatch_once_t;
typedef int hipError_t;
typedef void *hipFunction_t;
typedef void *hipModule_t;
typedef void *hipblasHandle_t;
typedef int hipblasStatus_t;
typedef void *hiprtcProgram;
typedef int hiprtcResult;
typedef float chelis_dynamic_name;
typedef unsigned short half;
typedef unsigned short __half;
typedef unsigned short hip_bfloat16;
typedef unsigned short bfloat;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef float float32x4_t __attribute__((ext_vector_type(4)));
typedef float __m256 __attribute__((vector_size(32)));
#define __device__
#define __global__
#define __host__
#define __shared__
#define __constant__
#define __forceinline__ inline
#define device
#define thread
#define threadgroup
#define constant
#ifdef __cplusplus
template <class T, int N = 0> struct vector {};
#endif
#ifdef __OBJC__
@class MPSMatrix;
@class MPSMatrixDescriptor;
@class MPSMatrixMultiplication;
@class NSMutableDictionary;
@class NSError;
@class NSString;
@protocol MTLBuffer;
#else
typedef void *MPSMatrix;
typedef void *MPSMatrixDescriptor;
typedef void *MPSMatrixMultiplication;
typedef void *MTLBuffer;
typedef void *NSMutableDictionary;
typedef void *NSError;
typedef void *NSString;
#endif
"#;

const MAX_PREPROCESSOR_CONFIGURATIONS: usize = 256;

#[derive(Clone, Debug)]
struct PreprocessorToken {
    spelling: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct PreprocessorDirective {
    start: usize,
    end: usize,
    tokens: Vec<PreprocessorToken>,
}

#[derive(Clone, Copy, Debug, Default)]
struct MacroUsage {
    definedness: bool,
    value: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct HeaderQuery {
    spelling: String,
}

#[derive(Clone, Debug)]
struct HeaderQueryUse {
    query: HeaderQuery,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MacroState {
    Undefined,
    Zero,
    One,
}

impl MacroState {
    fn label(self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::Zero => "0",
            Self::One => "1",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct CompilerConfiguration {
    macros: BTreeMap<String, MacroState>,
    present_headers: BTreeSet<HeaderQuery>,
}

impl CompilerConfiguration {
    fn label(&self, headers: &BTreeSet<HeaderQuery>) -> String {
        let mut entries = self
            .macros
            .iter()
            .map(|(name, state)| format!("{name}={}", state.label()))
            .collect::<Vec<_>>();
        entries.extend(headers.iter().map(|header| {
            let state = if self.present_headers.contains(header) {
                "present"
            } else {
                "absent"
            };
            format!("{}={state}", header.spelling)
        }));
        entries.join(", ")
    }
}

#[derive(Clone, Debug)]
struct PreprocessorBoundary {
    directives: Vec<PreprocessorDirective>,
    macro_usage: BTreeMap<String, MacroUsage>,
    headers: BTreeSet<HeaderQuery>,
    header_uses: Vec<HeaderQueryUse>,
}

fn compiler_base_arguments(path: &Path) -> Vec<String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let base = if path.to_string_lossy().contains("backend-metal")
        || matches!(extension, "m" | "mm" | "metal")
    {
        vec![
            "-xobjective-c++",
            "-std=gnu++17",
            "-fblocks",
            "-Drestrict=__restrict",
            "-D__cplusplus=201703L",
        ]
    } else if matches!(extension, "c") {
        vec!["-xc", "-std=gnu11"]
    } else if matches!(extension, "cu" | "cuh") {
        vec!["-xcuda", "-std=gnu++17", "-nocudainc", "-nocudalib"]
    } else if matches!(extension, "hip") {
        vec!["-xhip", "-std=gnu++17", "-nogpuinc", "-nogpulib"]
    } else {
        vec![
            "-xobjective-c++",
            "-std=gnu++17",
            "-fblocks",
            "-Drestrict=__restrict",
            "-D__cplusplus=201703L",
        ]
    };
    base.into_iter().map(str::to_string).collect()
}

fn compiler_arguments(path: &Path, configuration: &CompilerConfiguration) -> Vec<String> {
    let mut arguments = compiler_base_arguments(path);
    arguments.push("-undef".to_string());
    arguments.push("-nostdinc".to_string());
    for (name, state) in &configuration.macros {
        arguments.push(format!("-U{name}"));
        match state {
            MacroState::Undefined => {}
            MacroState::Zero => arguments.push(format!("-D{name}=0")),
            MacroState::One => arguments.push(format!("-D{name}=1")),
        }
    }
    arguments
}

fn preprocessing_token_spelling(spelling: &str) -> String {
    // C translation phase two deletes each backslash-newline splice before
    // preprocessing tokens are interpreted. Libclang's range tokenizer retains
    // the physical LF or CRLF bytes when a splice falls inside some tokens, so
    // establish the logical token spelling once at this boundary while leaving
    // the physical source offsets untouched for later materialization.
    spelling.replace("\\\r\n", "").replace("\\\n", "")
}

fn preprocessing_directives(
    index: &Index<'_>,
    source: &str,
    path: &Path,
) -> Result<Vec<PreprocessorDirective>, ScanError> {
    let virtual_path = Path::new("/__chelis_c_surface__/discovery").join(
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("source.cpp"),
    );
    let unsaved = [Unsaved::new(&virtual_path, source)];
    let mut arguments = compiler_base_arguments(path);
    arguments.extend(["-undef".to_string(), "-nostdinc".to_string()]);
    let mut parser = index.parser(&virtual_path);
    parser
        .arguments(&arguments)
        .unsaved(&unsaved)
        .detailed_preprocessing_record(true);
    let translation_unit = parser.parse().map_err(|error| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            format!("libclang could not tokenize C-family preprocessing directives: {error:?}"),
        )
    })?;
    let file = translation_unit.get_file(&virtual_path).ok_or_else(|| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "libclang did not retain the C-family source used for preprocessing discovery",
        )
    })?;
    let source_len = u32::try_from(source.len()).map_err(|_| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "C-family source exceeds libclang's 32-bit source-offset boundary",
        )
    })?;
    let range = SourceRange::new(
        file.get_offset_location(0),
        file.get_offset_location(source_len),
    );
    let clang_tokens = range.tokenize();
    let annotations = translation_unit.annotate(&clang_tokens);
    let mut directive_ranges = BTreeSet::new();
    for (token, entity) in clang_tokens.iter().zip(&annotations) {
        let Some(entity) = entity.filter(|entity| {
            entity.is_in_main_file()
                && matches!(
                    entity.get_kind(),
                    EntityKind::PreprocessingDirective | EntityKind::InclusionDirective
                )
        }) else {
            continue;
        };
        if !matches!(token.get_spelling().as_str(), "#" | "%:") {
            continue;
        }
        let Some(range) = entity.get_range() else {
            continue;
        };
        let start = range.get_start().get_spelling_location().offset as usize;
        let end = range.get_end().get_spelling_location().offset as usize;
        directive_ranges.insert((start, end));
    }

    Ok(directive_ranges
        .into_iter()
        .map(|(start, end)| {
            let tokens = clang_tokens
                .iter()
                .filter_map(|token| {
                    let range = token.get_range();
                    let token_start = range.get_start().get_spelling_location().offset as usize;
                    let token_end = range.get_end().get_spelling_location().offset as usize;
                    (token_start >= start && token_end <= end).then(|| PreprocessorToken {
                        spelling: preprocessing_token_spelling(&token.get_spelling()),
                        start: token_start,
                        end: token_end,
                    })
                })
                .collect();
            PreprocessorDirective { start, end, tokens }
        })
        .collect())
}

fn directive_keyword(directive: &PreprocessorDirective) -> Option<&str> {
    let introducer = directive.tokens.first()?.spelling.as_str();
    if !matches!(introducer, "#" | "%:") {
        return None;
    }
    directive.tokens.get(1).map(|token| token.spelling.as_str())
}

fn fixed_configuration_macro(name: &str) -> bool {
    matches!(name, "__cplusplus" | "__OBJC__" | "__has_include")
}

struct ConditionParser<'a> {
    tokens: &'a [PreprocessorToken],
    index: usize,
    usage: BTreeMap<String, MacroUsage>,
    headers: Vec<HeaderQueryUse>,
}

impl<'a> ConditionParser<'a> {
    fn new(tokens: &'a [PreprocessorToken]) -> Self {
        Self {
            tokens,
            index: 0,
            usage: BTreeMap::new(),
            headers: Vec::new(),
        }
    }

    fn spelling(&self) -> Option<&str> {
        self.tokens
            .get(self.index)
            .map(|token| token.spelling.as_str())
    }

    fn consume(&mut self, spelling: &str) -> bool {
        if self.spelling() == Some(spelling) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn expected(&self, expected: &str) -> ScanError {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            format!(
                "unsupported preprocessor condition: expected {expected}, found {}",
                self.spelling().unwrap_or("end of directive")
            ),
        )
    }

    fn mark_definedness(&mut self, name: &str) {
        if !fixed_configuration_macro(name) {
            self.usage.entry(name.to_string()).or_default().definedness = true;
        }
    }

    fn mark_value(&mut self, name: &str) {
        if !fixed_configuration_macro(name) {
            self.usage.entry(name.to_string()).or_default().value = true;
        }
    }

    fn parse(mut self) -> Result<(BTreeMap<String, MacroUsage>, Vec<HeaderQueryUse>), ScanError> {
        self.parse_or()?;
        if self.index != self.tokens.len() {
            return Err(self.expected("end of Boolean expression"));
        }
        Ok((self.usage, self.headers))
    }

    fn parse_or(&mut self) -> Result<(), ScanError> {
        self.parse_and()?;
        while matches!(self.spelling(), Some("||" | "or")) {
            self.index += 1;
            self.parse_and()?;
        }
        Ok(())
    }

    fn parse_and(&mut self) -> Result<(), ScanError> {
        self.parse_unary()?;
        while matches!(self.spelling(), Some("&&" | "and")) {
            self.index += 1;
            self.parse_unary()?;
        }
        Ok(())
    }

    fn parse_unary(&mut self) -> Result<(), ScanError> {
        if matches!(self.spelling(), Some("!" | "not")) {
            self.index += 1;
            self.parse_unary()
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<(), ScanError> {
        if self.consume("(") {
            self.parse_or()?;
            if !self.consume(")") {
                return Err(self.expected("`)`"));
            }
            return Ok(());
        }
        if self.consume("0") || self.consume("1") {
            return Ok(());
        }
        if self.consume("defined") {
            let parenthesized = self.consume("(");
            let name = self
                .spelling()
                .filter(|name| is_identifier(name))
                .ok_or_else(|| self.expected("a macro name after `defined`"))?
                .to_string();
            self.index += 1;
            if parenthesized && !self.consume(")") {
                return Err(self.expected("`)` after the macro name"));
            }
            self.mark_definedness(&name);
            return Ok(());
        }
        if self.spelling() == Some("__has_include") {
            return self.parse_has_include();
        }
        let name = self
            .spelling()
            .filter(|name| is_identifier(name))
            .ok_or_else(|| self.expected("a Boolean macro, `defined`, or `__has_include`"))?
            .to_string();
        self.index += 1;
        self.mark_value(&name);
        Ok(())
    }

    fn parse_has_include(&mut self) -> Result<(), ScanError> {
        let start = self
            .tokens
            .get(self.index)
            .expect("caller established the __has_include token")
            .start;
        self.index += 1;
        if !self.consume("(") {
            return Err(self.expected("`(` after `__has_include`"));
        }
        let (spelling, path) = if self.consume("<") {
            let mut path = String::new();
            while let Some(spelling) = self.spelling() {
                if spelling == ">" {
                    break;
                }
                path.push_str(spelling);
                self.index += 1;
            }
            if path.is_empty() || !self.consume(">") {
                return Err(self.expected("a literal `<header>`"));
            }
            (format!("<{path}>"), path)
        } else {
            let literal = self
                .spelling()
                .filter(|spelling| spelling.starts_with('"') && spelling.ends_with('"'))
                .ok_or_else(|| self.expected("a literal quoted header"))?
                .to_string();
            self.index += 1;
            let path = literal[1..literal.len() - 1].to_string();
            (literal, path)
        };
        if !self.consume(")") {
            return Err(self.expected("`)` after the literal header"));
        }
        let end = self.tokens[self.index - 1].end;
        if path.is_empty() {
            return Err(ScanError::new(
                ScanErrorKind::MalformedCandidate,
                format!("empty literal header name in `__has_include`: {spelling}"),
            ));
        }
        self.headers.push(HeaderQueryUse {
            query: HeaderQuery { spelling },
            start,
            end,
        });
        Ok(())
    }
}

fn merge_macro_usage(
    target: &mut BTreeMap<String, MacroUsage>,
    discovered: BTreeMap<String, MacroUsage>,
) {
    for (name, usage) in discovered {
        let entry = target.entry(name).or_default();
        entry.definedness |= usage.definedness;
        entry.value |= usage.value;
    }
}

fn preprocessor_boundary(
    index: &Index<'_>,
    source: &str,
    path: &Path,
) -> Result<PreprocessorBoundary, ScanError> {
    let directives = preprocessing_directives(index, source, path)?;
    let mut macro_usage = BTreeMap::new();
    let mut headers = BTreeSet::new();
    let mut header_uses = Vec::new();

    for directive in &directives {
        let Some(keyword) = directive_keyword(directive) else {
            continue;
        };
        match keyword {
            "ifdef" | "ifndef" | "elifdef" | "elifndef" => {
                let arguments = &directive.tokens[2..];
                if arguments.len() != 1 || !is_identifier(&arguments[0].spelling) {
                    return Err(ScanError::new(
                        ScanErrorKind::MalformedCandidate,
                        format!("unsupported `#{keyword}` configuration directive"),
                    ));
                }
                if !fixed_configuration_macro(&arguments[0].spelling) {
                    macro_usage
                        .entry(arguments[0].spelling.clone())
                        .or_insert(MacroUsage::default())
                        .definedness = true;
                }
            }
            "if" | "elif" => {
                let (usage, queries) = ConditionParser::new(&directive.tokens[2..]).parse()?;
                merge_macro_usage(&mut macro_usage, usage);
                headers.extend(queries.iter().map(|query| query.query.clone()));
                header_uses.extend(queries);
            }
            _ => {}
        }
    }

    for directive in &directives {
        if directive_keyword(directive) != Some("define") {
            continue;
        }
        let Some(name) = directive.tokens.get(2) else {
            return Err(ScanError::new(
                ScanErrorKind::MalformedCandidate,
                "malformed `#define` preprocessing directive",
            ));
        };
        let Some(usage) = macro_usage.get(&name.spelling) else {
            continue;
        };
        let replacement = &directive.tokens[3..];
        let is_boolean = replacement.is_empty()
            || matches!(replacement, [PreprocessorToken { spelling, .. }] if spelling == "0" || spelling == "1");
        let empty_value_expression = replacement.is_empty() && usage.value;
        if !is_boolean || empty_value_expression {
            return Err(ScanError::new(
                ScanErrorKind::MalformedCandidate,
                format!(
                    "configuration macro `{}` has a source definition outside the closed empty/0/1 Boolean domain",
                    name.spelling
                ),
            ));
        }
    }

    Ok(PreprocessorBoundary {
        directives,
        macro_usage,
        headers,
        header_uses,
    })
}

fn compiler_configurations(
    macro_usage: &BTreeMap<String, MacroUsage>,
    headers: &BTreeSet<HeaderQuery>,
) -> Result<Vec<CompilerConfiguration>, ScanError> {
    let mut dimension_sizes = macro_usage
        .values()
        .map(|usage| {
            if usage.definedness && usage.value {
                3
            } else {
                2
            }
        })
        .collect::<Vec<_>>();
    dimension_sizes.extend(std::iter::repeat_n(2, headers.len()));
    let count = dimension_sizes.into_iter().try_fold(1usize, |count, size| {
        count
            .checked_mul(size)
            .filter(|next| *next <= MAX_PREPROCESSOR_CONFIGURATIONS)
            .ok_or(())
    });
    if count.is_err() {
        let mut dimensions = macro_usage.keys().cloned().collect::<Vec<_>>();
        dimensions.extend(headers.iter().map(|header| header.spelling.clone()));
        return Err(ScanError::new(
            ScanErrorKind::MalformedCandidate,
            format!(
                "C-family preprocessing configuration exceeds the {MAX_PREPROCESSOR_CONFIGURATIONS}-configuration cap: {}",
                dimensions.join(", ")
            ),
        ));
    }

    let mut configurations = vec![CompilerConfiguration::default()];
    for (name, usage) in macro_usage {
        let states: &[MacroState] = if usage.definedness && usage.value {
            &[MacroState::Undefined, MacroState::Zero, MacroState::One]
        } else {
            &[MacroState::Undefined, MacroState::One]
        };
        configurations = configurations
            .into_iter()
            .flat_map(|configuration| {
                states.iter().map(move |state| {
                    let mut next = configuration.clone();
                    next.macros.insert(name.clone(), *state);
                    next
                })
            })
            .collect();
    }
    for header in headers {
        configurations = configurations
            .into_iter()
            .flat_map(|configuration| {
                [false, true].into_iter().map(move |present| {
                    let mut next = configuration.clone();
                    if present {
                        next.present_headers.insert(header.clone());
                    }
                    next
                })
            })
            .collect();
    }
    Ok(configurations)
}

fn neutralize_nonsemantic_directives(
    source: &str,
    directives: &[PreprocessorDirective],
) -> Result<String, ScanError> {
    let mut bytes = source.as_bytes().to_vec();
    for directive in directives {
        if !matches!(
            directive_keyword(directive),
            Some("include" | "include_next" | "import" | "error")
        ) {
            continue;
        }
        if directive.start > directive.end || directive.end > bytes.len() {
            return Err(ScanError::new(
                ScanErrorKind::MalformedCandidate,
                "libclang returned a preprocessing directive outside the main source extent",
            ));
        }
        for byte in &mut bytes[directive.start..directive.end] {
            if !matches!(*byte, b'\n' | b'\r') {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(bytes).map_err(|_| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "preprocessing directive neutralization did not preserve UTF-8",
        )
    })
}

fn materialize_header_queries(
    source: &str,
    query_uses: &[HeaderQueryUse],
    configuration: &CompilerConfiguration,
) -> Result<String, ScanError> {
    let mut bytes = source.as_bytes().to_vec();
    for query_use in query_uses {
        if query_use.start >= query_use.end || query_use.end > bytes.len() {
            return Err(ScanError::new(
                ScanErrorKind::MalformedCandidate,
                "libclang returned an __has_include query outside the main source extent",
            ));
        }
        for index in query_use.start..query_use.end {
            let preserves_line_splice =
                bytes[index] == b'\\' && matches!(bytes.get(index + 1), Some(b'\n' | b'\r'));
            if !matches!(bytes[index], b'\n' | b'\r') && !preserves_line_splice {
                bytes[index] = b' ';
            }
        }
        bytes[query_use.start] = if configuration.present_headers.contains(&query_use.query) {
            b'1'
        } else {
            b'0'
        };
    }
    String::from_utf8(bytes).map_err(|_| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "header-query materialization did not preserve UTF-8",
        )
    })
}

fn materialize_format_holes(source: &str) -> Result<String, ScanError> {
    let chars: Vec<char> = source.chars().collect();
    let mut output = String::with_capacity(source.len());
    let mut index = 0usize;
    while index < chars.len() {
        if chars[index] != '{' {
            if chars[index] == '}' && chars.get(index + 1) == Some(&'}') {
                output.push('}');
                index += 2;
                continue;
            }
            output.push(chars[index]);
            index += 1;
            continue;
        }
        if chars.get(index + 1) == Some(&'{') {
            output.push('{');
            index += 2;
            continue;
        }
        let Some(close) = chars[index + 1..]
            .iter()
            .position(|character| *character == '}')
            .map(|offset| index + offset + 1)
        else {
            output.push('{');
            index += 1;
            continue;
        };
        let contents: String = chars[index + 1..close].iter().collect();
        let format_name = contents
            .split_once(':')
            .map_or(contents.trim(), |(name, _)| name.trim());
        if !contents.is_empty() && !is_identifier(format_name) {
            output.push('{');
            index += 1;
            continue;
        }
        let before = output.trim_end();
        let after: String = chars[close + 1..].iter().take(16).collect();
        let after_trimmed = after.trim_start();
        let starts_statement = before
            .rsplit(['{', ';'])
            .next()
            .is_some_and(|prefix| prefix.trim().is_empty());
        let after_starts_identifier = after_trimmed
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic() || character == '_');
        let replacement = if contents.contains(':') {
            if before.ends_with("0x") { "0000" } else { "0" }
        } else if after_starts_identifier
            && after
                .chars()
                .next()
                .is_some_and(|character| !character.is_whitespace())
        {
            ""
        } else if before.ends_with('[') && after_trimmed.starts_with(']') {
            "1"
        } else if after_trimmed.starts_with('.')
            || after_trimmed.starts_with(|character: char| character.is_ascii_digit())
        {
            "0"
        } else if before.ends_with(';') && after_trimmed.is_empty() {
            ""
        } else if before.ends_with('*') || before.ends_with('&') {
            "chelis_dynamic_name"
        } else if starts_statement && after_starts_identifier {
            "float"
        } else if after_trimmed.starts_with('(') {
            "chelis_dynamic_name"
        } else if after_trimmed.starts_with(['*', '&'])
            || after_trimmed.starts_with("const")
            || before.ends_with("sizeof(")
        {
            "float"
        } else if before.ends_with('(') && after_trimmed.starts_with(')') {
            let call_prefix = before[..before.len() - 1].trim_end();
            if call_prefix
                .chars()
                .last()
                .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
            {
                "0"
            } else {
                "float"
            }
        } else {
            "0"
        };
        output.push_str(replacement);
        index = close + 1;
    }
    Ok(output)
}

fn declaration_tokens(entity: Entity<'_>) -> String {
    entity
        .get_range()
        .map(|range| {
            range
                .tokenize()
                .into_iter()
                .map(|token| token.get_spelling())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

const INTERNAL_SOURCE_RANGE_MARKER: &str = ";__source-range=";

fn internal_source_range(entity: Entity<'_>) -> String {
    let Some(range) = entity.get_range() else {
        return "0..0".to_string();
    };
    let start = range.get_start().get_expansion_location().offset;
    let end = range.get_end().get_expansion_location().offset;
    format!("{start}..{end}")
}

fn is_numeric_type_spelling(spelling: &str) -> bool {
    let words = tokens(spelling);
    words.iter().any(|word| {
        ELEMENT_C_TYPES.contains(&word.as_str())
            || matches!(word.as_str(), "_Float16" | "__fp16" | "__bf16" | "__int128")
    })
}

fn has_carrier_shape(ty: Type<'_>) -> bool {
    let canonical = ty.get_canonical_type();
    matches!(
        canonical.get_kind(),
        TypeKind::Pointer
            | TypeKind::BlockPointer
            | TypeKind::MemberPointer
            | TypeKind::LValueReference
            | TypeKind::RValueReference
            | TypeKind::ConstantArray
            | TypeKind::DependentSizedArray
            | TypeKind::IncompleteArray
            | TypeKind::VariableArray
    ) || canonical.get_pointee_type().is_some_and(has_carrier_shape)
        || canonical.get_element_type().is_some_and(has_carrier_shape)
}

fn type_is_numeric_carrier(entity: Entity<'_>, ty: Type<'_>, source: &str) -> bool {
    let authored = authored_type_identity(entity, ty, source);
    let exposed = ty.get_display_name();
    if [&authored, &exposed].into_iter().any(|spelling| {
        tokens(spelling)
            .iter()
            .any(|word| EXTERNAL_NONNUMERIC_C_TYPES.contains(&word.as_str()))
    }) || (tokens(&authored).iter().any(|word| word == "char")
        && !tokens(&authored).iter().any(|word| {
            matches!(
                word.as_str(),
                "int8_t" | "uint8_t" | "int_fast8_t" | "uint_fast8_t"
            )
        }))
    {
        return false;
    }
    let canonical = ty.get_canonical_type();
    has_carrier_shape(canonical) && is_numeric_type_spelling(&canonical.get_display_name())
}

fn declaration_shape(ty: Type<'_>) -> &'static str {
    match ty.get_canonical_type().get_kind() {
        TypeKind::Pointer | TypeKind::BlockPointer | TypeKind::MemberPointer => "pointer",
        TypeKind::LValueReference => "lvalue-reference",
        TypeKind::RValueReference => "rvalue-reference",
        TypeKind::ConstantArray
        | TypeKind::DependentSizedArray
        | TypeKind::IncompleteArray
        | TypeKind::VariableArray => "array",
        _ => "carrier",
    }
}

fn entity_source(entity: Entity<'_>, source: &str) -> String {
    let Some(range) = entity.get_range() else {
        return String::new();
    };
    let start = range.get_start().get_expansion_location().offset as usize;
    let end = range.get_end().get_expansion_location().offset as usize;
    source.get(start..end).unwrap_or_default().to_string()
}

fn declarator_details(entity: Entity<'_>, source: &str) -> String {
    let tokens = entity_source(entity, source);
    let mut details = Vec::new();
    let mut chars = tokens.chars();
    while let Some(character) = chars.next() {
        if character == '[' {
            let mut extent = String::from("[");
            for inner in chars.by_ref() {
                extent.push(inner);
                if inner == ']' {
                    break;
                }
            }
            details.push(extent);
        }
    }
    details.join("")
}

fn authored_type_identity(entity: Entity<'_>, ty: Type<'_>, source: &str) -> String {
    if entity.get_kind() == EntityKind::TypedefDecl {
        return entity.get_name().unwrap_or_else(|| ty.get_display_name());
    }
    let declaration = entity_source(entity, source);
    let name = entity.get_name().unwrap_or_default();
    let mut words = Vec::new();
    for token in tokens(&declaration) {
        if matches!(token.as_str(), "*" | "&" | "[") || token == name {
            break;
        }
        if is_identifier(&token)
            && !QUALIFIERS.contains(&token.as_str())
            && !is_storage_specifier(&token)
        {
            words.push(token);
        }
    }
    if words.is_empty() {
        ty.get_display_name()
    } else {
        words.join(" ")
    }
}

fn compiler_signature(
    entity: Entity<'_>,
    ty: Type<'_>,
    owner: &str,
    source: &str,
    authored_context: Option<&str>,
) -> String {
    let canonical = ty.get_canonical_type();
    let name = entity
        .get_name()
        .unwrap_or_else(|| "<abstract>".to_string());
    let context = if entity.is_declaration() {
        "declaration"
    } else if entity.is_expression() {
        "expression"
    } else {
        "typed-entity"
    };
    let mut signature = format!(
        "context={context};ast={:?};shape={};resolved={};authored={};name={name};declarator={};enclosing={owner}",
        entity.get_kind(),
        declaration_shape(ty),
        canonical.get_display_name(),
        authored_type_identity(entity, ty, source),
        declarator_details(entity, source),
    );
    if entity.is_expression() {
        signature.push_str(&format!(";expression={}", declaration_tokens(entity)));
    }
    if let Some(context) = authored_context {
        signature.push_str(";emitted=");
        signature.push_str(&emitted_context_identity(context));
    }
    signature.push_str(INTERNAL_SOURCE_RANGE_MARKER);
    signature.push_str(&internal_source_range(entity));
    signature
}

fn compiler_return_signature(
    entity: Entity<'_>,
    return_type: Type<'_>,
    owner: &str,
    source: &str,
    authored_context: Option<&str>,
) -> String {
    let canonical_return = return_type.get_canonical_type();
    let canonical_function = entity
        .get_type()
        .map(|ty| ty.get_canonical_type().get_display_name())
        .unwrap_or_else(|| declaration_tokens(entity));
    let name = entity
        .get_name()
        .unwrap_or_else(|| "<abstract>".to_string());
    let mut signature = format!(
        "shape=return-{};resolved={canonical_function};return={};authored={};name={name};declarator={};enclosing={owner}",
        declaration_shape(return_type),
        canonical_return.get_display_name(),
        authored_type_identity(entity, return_type, source),
        declarator_details(entity, source),
    );
    if let Some(context) = authored_context {
        signature.push_str(";emitted=");
        signature.push_str(&emitted_context_identity(context));
    }
    signature.push_str(INTERNAL_SOURCE_RANGE_MARKER);
    signature.push_str(&internal_source_range(entity));
    signature
}

fn emitted_context_identity(context: &str) -> String {
    let source_tokens = tokens(context);
    let mut output = Vec::new();
    let mut index = 0usize;
    while index < source_tokens.len() {
        if source_tokens.get(index).map(String::as_str) == Some("{")
            && source_tokens.get(index + 2).map(String::as_str) == Some("}")
            && source_tokens
                .get(index + 1)
                .is_some_and(|token| is_identifier(token))
        {
            output.push(format!("format-hole:{}", source_tokens[index + 1]));
            index += 3;
        } else if source_tokens.get(index).map(String::as_str) == Some("{")
            && source_tokens.get(index + 1).map(String::as_str) == Some("}")
        {
            output.push("format-hole:positional".to_string());
            index += 2;
        } else {
            output.push(source_tokens[index].clone());
            index += 1;
        }
    }
    output.join(" ")
}

fn sizeof_resolved_type(spelling: &str) -> String {
    let source_tokens = tokens(spelling);
    source_tokens
        .iter()
        .filter(|word| ELEMENT_C_TYPES.contains(&word.as_str()) || is_base_modifier(word.as_str()))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn structural_owner_name(entity: Entity<'_>) -> Option<String> {
    let name = entity.get_display_name().or_else(|| entity.get_name())?;
    if name.contains("(unnamed ") || name.contains("(anonymous ") {
        Some(format!("<anonymous:{}>", declaration_tokens(entity)))
    } else {
        Some(name)
    }
}

fn structural_owner_path(entity: Entity<'_>, source_owner: &str) -> Option<String> {
    let mut names = Vec::new();
    let mut current = Some(entity);
    while let Some(candidate) = current {
        let parent = candidate
            .get_semantic_parent()
            .or_else(|| candidate.get_lexical_parent());
        if candidate.get_kind() == EntityKind::TranslationUnit || parent.is_none() {
            break;
        }
        if candidate.is_declaration()
            && candidate.get_name().as_deref() != Some("__chelis_surface_fragment")
            && let Some(name) = structural_owner_name(candidate)
        {
            names.push(name);
        }
        current = parent;
    }
    if names.is_empty() {
        None
    } else {
        names.reverse();
        Some(format!("{source_owner}::{}", names.join("::")))
    }
}

fn width_arithmetic_row(
    entity: Entity<'_>,
    owner: &str,
    authored_context: Option<&str>,
) -> Option<CarrierUse> {
    let spelling = declaration_tokens(entity);
    let spelling_tokens = tokens(&spelling);
    if spelling_tokens.first().map(String::as_str) != Some("sizeof") {
        return None;
    }
    let (shape, open) = if spelling_tokens.get(1).map(String::as_str) == Some("...") {
        ("sizeof-pack", 2)
    } else if spelling_tokens.get(1).map(String::as_str) == Some("(") {
        ("sizeof", 1)
    } else {
        return None;
    };
    if matching_close(&spelling_tokens, open, "(", ")").ok()? + 1 != spelling_tokens.len() {
        return None;
    }
    let emitted = authored_context
        .map(|context| format!(";emitted={}", emitted_context_identity(context)))
        .unwrap_or_default();
    Some(CarrierUse {
        kind: "width-arithmetic".to_string(),
        owner: owner.to_string(),
        signature: format!(
            "shape={shape};resolved={};declaration={spelling}{emitted}{INTERNAL_SOURCE_RANGE_MARKER}{}",
            sizeof_resolved_type(&spelling),
            internal_source_range(entity),
        ),
    })
}

fn callable_result_type(entity: Entity<'_>) -> Option<Type<'_>> {
    entity
        .get_type()
        .and_then(|ty| ty.get_result_type())
        .or_else(|| {
            entity
                .is_declaration()
                .then(|| entity.get_result_type())
                .flatten()
        })
}

fn project_compiler_entity(
    entity: Entity<'_>,
    source_owner: &str,
    owner: &str,
    source: &str,
    authored_context: Option<&str>,
) -> Result<(String, Vec<CarrierUse>), ScanError> {
    let in_main_file = entity
        .get_location()
        .is_some_and(|location| location.is_in_main_file());
    if in_main_file
        && (!entity.get_kind().is_valid() || entity.get_kind() == EntityKind::NotImplemented)
    {
        return Err(ScanError::new(
            ScanErrorKind::MalformedCandidate,
            format!(
                "libclang exposed an unsupported AST entity {:?}: {}",
                entity.get_kind(),
                declaration_tokens(entity)
            ),
        ));
    }
    let structural_owner = if in_main_file
        && entity.is_declaration()
        && entity.get_name().as_deref() != Some("__chelis_surface_fragment")
    {
        structural_owner_path(entity, source_owner).unwrap_or_else(|| owner.to_string())
    } else {
        owner.to_string()
    };
    let mut rows = Vec::new();

    if in_main_file
        && (entity.is_declaration() || entity.is_expression())
        && let Some(ty) = entity.get_type()
        && type_is_numeric_carrier(entity, ty, source)
    {
        rows.push(CarrierUse {
            kind: "raw-element-pointer".to_string(),
            owner: structural_owner.clone(),
            signature: compiler_signature(entity, ty, &structural_owner, source, authored_context),
        });
    }

    if in_main_file
        && let Some(return_type) = callable_result_type(entity)
        && type_is_numeric_carrier(entity, return_type, source)
    {
        rows.push(CarrierUse {
            kind: "raw-element-pointer".to_string(),
            owner: structural_owner.clone(),
            signature: compiler_return_signature(
                entity,
                return_type,
                &structural_owner,
                source,
                authored_context,
            ),
        });
    }

    if in_main_file
        && let Some(row) = width_arithmetic_row(entity, &structural_owner, authored_context)
    {
        rows.push(row);
    }

    Ok((structural_owner, rows))
}

fn walk_compiler_ast(
    entity: Entity<'_>,
    source_owner: &str,
    owner: &str,
    source: &str,
    authored_context: Option<&str>,
    rows: &mut Vec<CarrierUse>,
) -> Result<(), ScanError> {
    let (structural_owner, entity_rows) =
        project_compiler_entity(entity, source_owner, owner, source, authored_context)?;
    rows.extend(entity_rows);

    for child in entity.get_children() {
        walk_compiler_ast(
            child,
            source_owner,
            &structural_owner,
            source,
            authored_context,
            rows,
        )?;
    }
    Ok(())
}

fn preserve_structural_multiplicity(rows: &mut Vec<CarrierUse>) {
    rows.sort();
    rows.dedup();
    let mut occurrences = BTreeMap::new();
    for row in rows {
        let marker = row.signature.rfind(INTERNAL_SOURCE_RANGE_MARKER).expect(
            "compiler-produced carrier identity must retain its internal source range marker",
        );
        row.signature.truncate(marker);
        let key = (row.kind.clone(), row.owner.clone(), row.signature.clone());
        let occurrence = occurrences.entry(key).or_insert(0usize);
        row.signature.push_str(&format!(";occurrence={occurrence}"));
        *occurrence += 1;
    }
}

fn compiler_scan(
    source: &str,
    source_owner: &str,
    path: &Path,
    authored_context: Option<&str>,
    materialize_holes: bool,
    extra_prelude: Option<&str>,
) -> Result<Vec<CarrierUse>, ScanError> {
    let _guard = CLANG_SCAN_LOCK.lock().map_err(|_| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            "libclang scanner lock poisoned",
        )
    })?;
    let clang = Clang::new().map_err(|error| {
        ScanError::new(
            ScanErrorKind::MalformedCandidate,
            format!("cannot load libclang for C-family scan: {error}"),
        )
    })?;
    let index = Index::new(&clang, true, false);
    let virtual_path = Path::new("/__chelis_c_surface__/main").join(
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("chelis_surface.cpp"),
    );
    let prelude_path = Path::new("/__chelis_c_surface__/chelis_c_surface_prelude.hpp");
    let materialized = if materialize_holes {
        materialize_format_holes(source)?
    } else {
        source.to_string()
    };
    let boundary = preprocessor_boundary(&index, &materialized, path)?;
    let cxx = path.extension().and_then(|value| value.to_str()) != Some("c");
    let mut materialized = neutralize_nonsemantic_directives(&materialized, &boundary.directives)?;
    let trimmed = materialized.trim_end();
    if !trimmed.is_empty()
        && !trimmed.ends_with([';', '}', '{'])
        && has_candidate_anchor(&tokens(trimmed))
    {
        materialized.push(';');
    }
    let configurations = compiler_configurations(&boundary.macro_usage, &boundary.headers)?;
    let language_prefix = if cxx {
        "#ifndef __cplusplus\n#define __cplusplus 201703L\n#endif\n"
    } else {
        ""
    };
    let prelude = format!("{CLANG_PRELUDE}\n{}", extra_prelude.unwrap_or_default());
    let mut rows = Vec::new();
    for configuration in configurations {
        let arguments = compiler_arguments(path, &configuration);
        let configuration_label = configuration.label(&boundary.headers);
        let configured_source =
            materialize_header_queries(&materialized, &boundary.header_uses, &configuration)?;
        let main_source = format!(
            "{language_prefix}#include \"{}\"\n{}\n",
            prelude_path.display(),
            configured_source,
        );
        let unsaved = vec![
            Unsaved::new(&virtual_path, &main_source),
            Unsaved::new(prelude_path, &prelude),
        ];
        let mut parser = index.parser(&virtual_path);
        parser
            .arguments(&arguments)
            .unsaved(&unsaved)
            .detailed_preprocessing_record(true);
        let translation_unit = parser.parse().map_err(|error| {
            ScanError::new(
                ScanErrorKind::MalformedCandidate,
                format!(
                    "libclang could not parse C-family source under configuration [{}]: {error:?}",
                    configuration_label
                ),
            )
        })?;
        for diagnostic in translation_unit.get_diagnostics() {
            if diagnostic.get_severity() < Severity::Error
                || !diagnostic.get_location().is_in_main_file()
            {
                continue;
            }
            let message = diagnostic.get_text();
            let location = diagnostic.get_location().get_spelling_location();
            let source_line = main_source
                .lines()
                .nth(location.line.saturating_sub(1) as usize)
                .unwrap_or_default();
            let located_message = format!(
                "line {} column {} under configuration [{}]: {message}; source: {source_line:?}",
                location.line, location.column, configuration_label
            );
            if authored_context.is_none()
                && (message.contains("unknown type name")
                    || message.contains("no template named")
                    || message.contains("decimal type")
                    || message.contains("unknown type")
                    || source.contains("__new_"))
            {
                return Err(ScanError::new(ScanErrorKind::UnknownType, located_message));
            }
            if diagnostic.get_severity() == Severity::Fatal
                || (authored_context.is_none()
                    && (message.contains("expected")
                        || message.contains("unterminated")
                        || message.contains("extraneous"))
                    && !message.starts_with("expected method"))
            {
                return Err(ScanError::new(
                    ScanErrorKind::MalformedCandidate,
                    located_message,
                ));
            }
        }
        walk_compiler_ast(
            translation_unit.get_entity(),
            source_owner,
            source_owner,
            &main_source,
            authored_context,
            &mut rows,
        )?;
    }
    preserve_structural_multiplicity(&mut rows);
    rows.sort();
    rows.dedup();
    Ok(rows)
}

pub fn scan_c_source(source: &str, owner: &str) -> Result<Vec<CarrierUse>, ScanError> {
    if !has_candidate_anchor(&tokens(source)) {
        return Ok(Vec::new());
    }
    validate_supported_arithmetic_spelling(source)?;
    validate_c_lexical_closure(source)?;
    validate_balanced(&tokens(source))?;
    compiler_scan(source, owner, Path::new("fixture.cpp"), None, false, None)
}

fn validate_supported_arithmetic_spelling(source: &str) -> Result<(), ScanError> {
    if tokens(source).iter().any(|word| {
        word.starts_with("_Float")
            || word.starts_with("_Decimal")
            || word.starts_with("__fp")
            || word.starts_with("__bf")
            || word.starts_with("__int")
    }) {
        return Err(ScanError::new(
            ScanErrorKind::UnknownType,
            "unknown arithmetic spelling at a C carrier position",
        ));
    }
    Ok(())
}

fn aliases_as_compiler_prelude(aliases: &BTreeMap<String, Vec<String>>) -> String {
    let mut prelude = String::new();
    for (name, words) in aliases {
        let mut definitions: Vec<String> = words
            .iter()
            .filter_map(|word| decode_alias_definition(word))
            .collect();
        definitions.sort();
        definitions.dedup();
        if let Some(definition) = definitions
            .iter()
            .find(|definition| {
                let definition_tokens = tokens(definition);
                has_candidate_anchor(&definition_tokens)
                    && definition_tokens
                        .iter()
                        .any(|word| ELEMENT_C_TYPES.contains(&word.as_str()))
            })
            .or_else(|| definitions.first())
        {
            prelude.push_str(definition);
            prelude.push('\n');
            continue;
        }
        let resolved = resolve_words(words.clone(), aliases);
        let target = if resolved
            .iter()
            .any(|word| ELEMENT_C_TYPES.contains(&word.as_str()))
        {
            resolved
                .iter()
                .find(|word| ELEMENT_C_TYPES.contains(&word.as_str()))
                .map_or("float", String::as_str)
        } else {
            "void"
        };
        prelude.push_str(&format!("typedef {target} {name};\n"));
    }
    prelude
}

pub fn scan_c_source_with_aliases(
    source: &str,
    owner: &str,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<CarrierUse>, ScanError> {
    validate_supported_arithmetic_spelling(source)?;
    validate_c_lexical_closure(source)?;
    validate_balanced(&tokens(source))?;
    let prelude = aliases_as_compiler_prelude(aliases);
    compiler_scan(
        source,
        owner,
        Path::new("fixture.cpp"),
        None,
        false,
        Some(&prelude),
    )
}

pub fn scan_c_source_at_path(
    source: &str,
    owner: &str,
    path: &Path,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<CarrierUse>, ScanError> {
    validate_supported_arithmetic_spelling(source)?;
    validate_c_lexical_closure(source)?;
    validate_balanced(&tokens(source))?;
    let prelude = aliases_as_compiler_prelude(aliases);
    compiler_scan(source, owner, path, None, false, Some(&prelude))
}

fn scan_c_fragment(source: &str, owner: &str) -> Result<Vec<CarrierUse>, ScanError> {
    let source_tokens = tokens(source);
    if !has_candidate_anchor(&source_tokens) {
        return Ok(Vec::new());
    }
    match scan_source_tokens(source, owner, false, None) {
        Ok(rows) if rows.is_empty() && !source_tokens.iter().any(|token| token == "sizeof") => {
            return Ok(Vec::new());
        }
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind,
                ScanErrorKind::MalformedCandidate | ScanErrorKind::IncompleteFragment
            ) =>
        {
            return Err(error);
        }
        Err(_) => {}
    }
    let mut top_level = source.to_string();
    let open_braces =
        tokens(source)
            .into_iter()
            .fold(0isize, |depth, token| match token.as_str() {
                "{" => depth + 1,
                "}" => depth - 1,
                _ => depth,
            });
    if open_braces > 0 {
        top_level.push_str(&"}".repeat(open_braces as usize));
    }
    if let Ok(rows) = compiler_scan(
        &top_level,
        owner,
        Path::new("emitted.cpp"),
        Some(source),
        false,
        None,
    ) && !rows.is_empty()
    {
        return Ok(rows);
    }
    let wrapped = format!("void __chelis_surface_fragment(void) {{ {source}; }}");
    compiler_scan(
        &wrapped,
        owner,
        Path::new("emitted.cpp"),
        Some(source),
        false,
        None,
    )
}

fn scan_c_format_fragment(source: &str, owner: &str) -> Result<Vec<CarrierUse>, ScanError> {
    let source_tokens = tokens(source);
    if !has_candidate_anchor(&source_tokens) {
        return Ok(Vec::new());
    }
    match scan_source_tokens(source, owner, false, None) {
        Ok(rows) if rows.is_empty() && !source_tokens.iter().any(|token| token == "sizeof") => {
            return Ok(Vec::new());
        }
        Err(error) if error.kind == ScanErrorKind::IncompleteFragment => return Err(error),
        Ok(_) | Err(_) => {}
    }
    let wrapped = format!("void __chelis_surface_fragment(void) {{ {source}; }}");
    compiler_scan(
        &wrapped,
        owner,
        Path::new("emitted.cpp"),
        Some(source),
        true,
        None,
    )
}

#[derive(Default)]
struct RustStringScanner {
    owners: Vec<String>,
    rows: Vec<CarrierUse>,
    error: Option<ScanError>,
}

impl RustStringScanner {
    fn owner(&self) -> String {
        if self.owners.is_empty() {
            "module".to_string()
        } else {
            self.owners.join("::")
        }
    }

    fn scan_literal(&mut self, literal: &syn::LitStr) {
        if self.error.is_some() {
            return;
        }
        let value = literal.value();
        let trimmed = value.trim_start();
        if trimmed.starts_with('*')
            && trimmed
                .chars()
                .nth(1)
                .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        {
            self.error = Some(ScanError::new(
                ScanErrorKind::IncompleteFragment,
                "C carrier fragment starts with a declarator pointer but has no type",
            ));
            return;
        }
        let owner = self.owner();
        match scan_c_fragment(&value, &owner) {
            Ok(mut rows) => self.rows.append(&mut rows),
            Err(error) => {
                let excerpt: String = value.chars().take(120).collect();
                self.error = Some(ScanError::new(
                    error.kind,
                    format!(
                        "Rust string in `{}` failed C-surface parsing: {}; excerpt: {excerpt:?}",
                        owner, error.message
                    ),
                ));
            }
        }
    }

    fn with_owner(&mut self, owner: String, visit: impl FnOnce(&mut Self)) {
        self.owners.push(owner);
        visit(self);
        self.owners.pop();
    }

    fn scan_token_stream_literals(&mut self, stream: proc_macro2::TokenStream) {
        for token in stream {
            match token {
                proc_macro2::TokenTree::Group(group) => {
                    self.scan_token_stream_literals(group.stream());
                }
                proc_macro2::TokenTree::Literal(literal) => {
                    if let Ok(syn::Lit::Str(literal)) = syn::parse_str(&literal.to_string()) {
                        self.scan_literal(&literal);
                    }
                }
                proc_macro2::TokenTree::Ident(_) | proc_macro2::TokenTree::Punct(_) => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for RustStringScanner {
    fn visit_attribute(&mut self, _attribute: &'ast syn::Attribute) {}

    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        self.with_owner(function.sig.ident.to_string(), |scanner| {
            visit::visit_item_fn(scanner, function);
        });
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        self.with_owner(function.sig.ident.to_string(), |scanner| {
            visit::visit_impl_item_fn(scanner, function);
        });
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if !is_exact_cfg_test_module(module) {
            self.with_owner(module.ident.to_string(), |scanner| {
                visit::visit_item_mod(scanner, module);
            });
        }
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let owner = item.self_ty.to_token_stream().to_string();
        self.with_owner(owner, |scanner| visit::visit_item_impl(scanner, item));
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if matches!(&field.member, syn::Member::Named(name) if name == "data") {
            self.rows.push(CarrierUse {
                kind: "direct-data-access".to_string(),
                owner: self.owner(),
                signature: format!("rust-field={}", field.to_token_stream()),
            });
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_lit_str(&mut self, literal: &'ast syn::LitStr) {
        self.scan_literal(literal);
    }

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        let macro_name = macro_call
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string());
        if macro_name.as_deref() == Some("stringify") {
            let source = macro_call.tokens.to_string();
            match scan_c_fragment(&source, &self.owner()) {
                Ok(mut rows) => self.rows.append(&mut rows),
                Err(error) => self.error = Some(error),
            }
            return;
        }
        let parser = Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(arguments) = parser.parse2(macro_call.tokens.clone()) {
            if macro_name.as_deref() == Some("format")
                && let Some(syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(format_string),
                    ..
                })) = arguments.first()
            {
                let text = format_string.value();
                let without_holes = tokens(&text)
                    .into_iter()
                    .filter(|token| !matches!(token.as_str(), "{" | "}"))
                    .filter(|token| !is_identifier(token))
                    .collect::<Vec<_>>();
                if text.contains('{') && without_holes.is_empty() {
                    self.rows.push(CarrierUse {
                        kind: "raw-element-pointer".to_string(),
                        owner: self.owner(),
                        signature: format!(
                            "shape=unconstrained-emission;emitted={}",
                            emitted_context_identity(&text)
                        ),
                    });
                    return;
                }
                match scan_c_format_fragment(&text, &self.owner()) {
                    Ok(mut rows) => self.rows.append(&mut rows),
                    Err(error) => self.error = Some(error),
                }
                for argument in arguments.iter().skip(1) {
                    self.visit_expr(argument);
                }
                return;
            }
            for argument in &arguments {
                self.visit_expr(argument);
            }
        } else {
            self.scan_token_stream_literals(macro_call.tokens.clone());
        }
    }
}

fn is_exact_cfg_test_module(module: &syn::ItemMod) -> bool {
    module.attrs.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Path>()
                .is_ok_and(|path| path.is_ident("test"))
    })
}

#[derive(Default)]
struct TestModuleSpans {
    spans: Vec<proc_macro2::Span>,
}

impl<'ast> Visit<'ast> for TestModuleSpans {
    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_exact_cfg_test_module(module) {
            self.spans.push(module.span());
        } else {
            visit::visit_item_mod(self, module);
        }
    }
}

pub fn production_rust_source(source: &str) -> Result<String, ScanError> {
    let file = syn::parse_file(source).map_err(|error| {
        ScanError::new(
            ScanErrorKind::InvalidRust,
            format!("cannot parse Rust source before test-region exclusion: {error}"),
        )
    })?;
    let mut visitor = TestModuleSpans::default();
    visitor.visit_file(&file);
    let mut line_starts = vec![0usize];
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            line_starts.push(index + 1);
        }
    }
    let offset = |location: proc_macro2::LineColumn| -> Option<usize> {
        line_starts
            .get(location.line.checked_sub(1)?)
            .and_then(|start| start.checked_add(location.column))
    };
    let mut bytes = source.as_bytes().to_vec();
    for span in visitor.spans {
        let Some(start) = offset(span.start()) else {
            continue;
        };
        let Some(end) = offset(span.end()) else {
            continue;
        };
        for byte in bytes.get_mut(start..end).into_iter().flatten() {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(bytes).map_err(|error| {
        ScanError::new(
            ScanErrorKind::InvalidRust,
            format!("test-region exclusion damaged Rust UTF-8: {error}"),
        )
    })
}

pub fn scan_rust_source(source: &str) -> Result<Vec<CarrierUse>, ScanError> {
    let file = syn::parse_file(source).map_err(|error| {
        ScanError::new(
            ScanErrorKind::InvalidRust,
            format!("cannot parse Rust source before C-carrier scan: {error}"),
        )
    })?;
    let mut scanner = RustStringScanner::default();
    scanner.visit_file(&file);
    if let Some(error) = scanner.error {
        Err(error)
    } else {
        Ok(scanner.rows)
    }
}
