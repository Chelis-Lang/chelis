//! Shared fail-closed C-family source classification.
//!
//! The runtime-representation and capacity censuses use the same lexical
//! normalization, type vocabulary, and alias resolution.  Carrier discovery
//! is token- and structure-based: qualifier placement and whitespace never
//! decide whether a pointer, array, cast, or `sizeof(type)` is visible.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
        if current.is_ascii_alphanumeric() || current == '_' || current == '.' {
            let mut token = String::new();
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric()
                    || chars[index] == '_'
                    || chars[index] == '.')
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
            if let Some((name, target)) = function_pointer_typedef(&statement) {
                merge_alias_words(&mut aliases, name, target);
            }
            continue;
        }
        if rest.contains('[') {
            if let Some((name, target)) = array_typedef(&statement) {
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
                next.extend(target.iter().cloned());
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
    token
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
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
        let words: Vec<String> = tokens(replacement)
            .into_iter()
            .filter(|token| is_identifier(token))
            .collect();
        if !words.is_empty() {
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
                next.extend(target.iter().cloned());
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
    tokens
        .iter()
        .any(|token| token == "*" || token == "&" || token == "[" || token == "sizeof")
}

fn scan_source(
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

pub fn scan_c_source(source: &str, owner: &str) -> Result<Vec<CarrierUse>, ScanError> {
    scan_source(source, owner, true, None)
}

pub fn scan_c_source_with_aliases(
    source: &str,
    owner: &str,
    aliases: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<CarrierUse>, ScanError> {
    scan_source(source, owner, true, Some(aliases))
}

fn scan_c_fragment(source: &str, owner: &str) -> Result<Vec<CarrierUse>, ScanError> {
    scan_source(source, owner, false, None)
}

#[derive(Default)]
struct RustStringScanner {
    owners: Vec<String>,
    rows: Vec<CarrierUse>,
    error: Option<ScanError>,
}

impl RustStringScanner {
    fn owner(&self) -> &str {
        self.owners.last().map_or("module", String::as_str)
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
        match scan_c_fragment(&value, self.owner()) {
            Ok(mut rows) => self.rows.append(&mut rows),
            Err(error) => {
                let excerpt: String = value.chars().take(120).collect();
                self.error = Some(ScanError::new(
                    error.kind,
                    format!(
                        "Rust string in `{}` failed C-surface parsing: {}; excerpt: {excerpt:?}",
                        self.owner(),
                        error.message
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
        let is_test = module.attrs.iter().any(|attribute| {
            attribute.path().is_ident("cfg")
                && attribute
                    .parse_args::<syn::Path>()
                    .is_ok_and(|path| path.is_ident("test"))
        });
        if !is_test {
            visit::visit_item_mod(self, module);
        }
    }

    fn visit_lit_str(&mut self, literal: &'ast syn::LitStr) {
        self.scan_literal(literal);
    }

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        let parser = Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(arguments) = parser.parse2(macro_call.tokens.clone()) {
            for argument in &arguments {
                self.visit_expr(argument);
            }
        } else {
            self.scan_token_stream_literals(macro_call.tokens.clone());
        }
    }
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
