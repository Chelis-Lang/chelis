//! Shared lexical primitives for the numeric capacity census.
//!
//! `spec/design/dtype_semantics.md` section C6 owns these: the comment/literal
//! stripper, the canonical token identity, the closed non-arithmetic type-word
//! list, and the alias-resolving classifier. They live here rather than inline
//! in one tripwire so the header, wire, and binding legs share exactly one
//! implementation, and they deliberately depend on nothing but `std`.
//!
//! Identity is a token sequence, never a preprocessor's incidental whitespace:
//! Apple clang and Linux gcc print the same declaration differently, and the
//! census must produce one row for both.

#![allow(dead_code)]

use std::collections::BTreeMap;

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
    "auto",
    "bool",
    "char",
    "const",
    "enum",
    "extern",
    "inline",
    "noexcept",
    "register",
    "restrict",
    "static",
    "std",
    "struct",
    "typedef",
    "union",
    "void",
    "volatile",
    "wchar_t",
    "__restrict",
    "__restrict__",
];

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

pub fn lex_c_tokens(source: &str) -> Vec<String> {
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

const ALIAS_DEFINITION_PREFIX: &str = "__chelis_alias_definition_";

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
