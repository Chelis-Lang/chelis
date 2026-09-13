//! Phase 2 typed-consumer guard for chelis#730.
//!
//! This test is supporting evidence: rustc exhaustiveness is the authority.
//! Positive evidence belongs to each relevant consumer crate, so moving an
//! owner within that crate does not require an inventory edit. Known raw
//! bypass spellings are scanned recursively across the relevant production
//! source trees and published headers, so relocating one cannot evade the
//! negative guard.

use std::fs;
use std::path::{Path, PathBuf};

#[path = "../../../tests/support/c_lexical.rs"]
mod c_lexical;

struct CrateEvidence {
    crate_name: &'static str,
    role: &'static str,
    roots: &'static [&'static str],
    required: &'static [&'static str],
}

struct ForbiddenRule {
    role: &'static str,
    roots: &'static [&'static str],
    forbidden: &'static [&'static str],
}

struct ScannedSource {
    path: PathBuf,
    language: SourceLanguage,
    evidence_tokens: Vec<String>,
    active_tokens: Vec<String>,
}

impl ScannedSource {
    fn new(path: PathBuf, contents: &str) -> Self {
        let language = SourceLanguage::from_path(&path)
            .unwrap_or_else(|| panic!("unsupported scanned source extension: {}", path.display()));
        let tokens = language
            .lex(contents)
            .unwrap_or_else(|error| panic!("cannot lex {}: {error}", path.display()));
        let evidence_tokens = project_evidence(language, &tokens);
        let active_tokens = project_active(&tokens);
        Self {
            path,
            language,
            evidence_tokens,
            active_tokens,
        }
    }

    fn contains_evidence_pattern(&self, pattern: &str) -> bool {
        let tokens = self
            .language
            .lex(pattern)
            .unwrap_or_else(|error| panic!("invalid evidence pattern `{pattern}`: {error}"));
        contains_token_pattern(
            &self.evidence_tokens,
            &project_evidence(self.language, &tokens),
        )
    }

    fn contains_active_pattern(&self, pattern: &str) -> bool {
        let tokens = self
            .language
            .lex(pattern)
            .unwrap_or_else(|error| panic!("invalid forbidden pattern `{pattern}`: {error}"));
        contains_token_pattern(&self.active_tokens, &project_active(&tokens))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceLanguage {
    Rust,
    CFamily,
}

impl SourceLanguage {
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("rs") => Some(Self::Rust),
            Some("h" | "hh" | "hpp" | "hxx" | "cuh" | "c" | "cc" | "cpp" | "cu" | "m" | "mm") => {
                Some(Self::CFamily)
            }
            _ => None,
        }
    }

    fn lex(self, source: &str) -> Result<Vec<SourceToken>, String> {
        match self {
            Self::Rust => lex_rust_tokens(source),
            Self::CFamily => Ok(lex_c_family_tokens(source)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SourceTokenKind {
    Syntax,
    StringLiteral,
    CharacterLiteral,
    RawStringLiteral,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceToken {
    kind: SourceTokenKind,
    spelling: String,
}

impl SourceToken {
    fn syntax(spelling: impl Into<String>) -> Self {
        Self {
            kind: SourceTokenKind::Syntax,
            spelling: spelling.into(),
        }
    }

    fn literal(kind: SourceTokenKind, spelling: impl Into<String>) -> Self {
        Self {
            kind,
            spelling: spelling.into(),
        }
    }
}

const EFFECT_KIND_CRATES: &[CrateEvidence] = &[
    CrateEvidence {
        crate_name: "chelis-deep",
        role: "Deep metadata decode boundary",
        roots: &["crates/chelis-deep/src"],
        required: &["EffectKindInput", "Result<EffectKind"],
    },
    CrateEvidence {
        crate_name: "chelis-types",
        role: "checker handle-effect semantics",
        roots: &["crates/chelis-types/src"],
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
    },
    CrateEvidence {
        crate_name: "chelis-effects",
        role: "effect inference and handler/target validation",
        roots: &["crates/chelis-effects/src"],
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
    },
    CrateEvidence {
        crate_name: "chelis-ir",
        role: "IR and host lowering handle-effect semantics",
        roots: &["crates/chelis-ir/src"],
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
    },
    CrateEvidence {
        crate_name: "chelis-compiler-api",
        role: "compiler-api evaluator handle-effect semantics",
        roots: &["crates/chelis-compiler-api/src"],
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
    },
    CrateEvidence {
        crate_name: "chelis-surf",
        role: "Surf/Deep canonical effect conversion",
        roots: &["crates/chelis-surf/src"],
        required: &[
            "use chelis_vocab::EffectKind",
            "M::Effect",
            ".effect()",
            "decode_effect_kind(",
            "EffectKind::Random",
            "EffectKind::Resource",
        ],
    },
];

const RUNTIME_DTYPE_CRATES: &[CrateEvidence] = &[
    CrateEvidence {
        crate_name: "chelis-types",
        role: "Prim-to-runtime ABI adapter",
        roots: &["crates/chelis-types/src"],
        required: &[
            "use chelis_vocab::RuntimeDType",
            "runtime_dtype(self)",
            "Result<RuntimeDType",
        ],
    },
    CrateEvidence {
        crate_name: "chelis-runtime",
        role: "runtime ABI decode and semantic dispatch",
        roots: &["crates/chelis-runtime/src"],
        required: &[
            "RuntimeDType",
            "decode_runtime_dtype(dtype: chelis_dtype)",
            "tensor_elem_size(dtype: RuntimeDType)",
            "read_index_slot(",
            "dtype: RuntimeDType",
        ],
    },
    CrateEvidence {
        crate_name: "chelis-backend-c",
        role: "C codegen runtime dtype selection",
        roots: &["crates/chelis-backend-c/src"],
        required: &[
            ".runtime_dtype()",
            ".c_macro()",
            "prim.runtime_dtype()",
            "self.runtime_dtype().c_macro()",
        ],
    },
    CrateEvidence {
        crate_name: "chelis-backend-hip",
        role: "HIP codegen and checked device allocation",
        roots: &[
            "crates/chelis-backend-hip/src",
            "crates/chelis-backend-hip/runtime",
        ],
        required: &[
            ".runtime_dtype()",
            ".c_macro()",
            "#include \"chelis_runtime_dtype.h\"",
            "#include \"chelis_device_owner.h\"",
            "auto bytes = chelis_metadata_plan_byte_count(plan.get());",
            "hipMalloc(&data, static_cast<size_t>(bytes))",
        ],
    },
    CrateEvidence {
        crate_name: "chelis-abi",
        role: "checked metadata derives bytes from RuntimeDType",
        roots: &["crates/chelis-abi/src"],
        required: &[
            "use chelis_vocab::RuntimeDType",
            "pub fn bytes(self, dtype: RuntimeDType)",
            "self.layout_bytes(dtype.contract().repr().byte_width())",
        ],
    },
    CrateEvidence {
        crate_name: "chelis-backend-metal",
        role: "Metal runtime dtype tag selection",
        roots: &["crates/chelis-backend-metal/src"],
        required: &["prec.runtime_dtype()", ".c_macro()"],
    },
    CrateEvidence {
        crate_name: "chelis-python",
        role: "Python FFI dtype decode and constants",
        roots: &["crates/chelis-python/src"],
        required: &["RuntimeDType::F32.id()", "decode_runtime_dtype("],
    },
];

const EFFECT_KIND_FORBIDDEN_RULES: &[ForbiddenRule] = &[
    ForbiddenRule {
        role: "Deep metadata decode boundary",
        roots: &["crates/chelis-deep/src"],
        forbidden: &["Option<EffectKind"],
    },
    ForbiddenRule {
        role: "checker handle-effect semantics",
        roots: &["crates/chelis-types/src"],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    ForbiddenRule {
        role: "effect inference and handler/target validation",
        roots: &["crates/chelis-effects/src"],
        forbidden: &[
            "match effect_name(list)",
            "effect_name(list) == Some(\"resource\")",
        ],
    },
    ForbiddenRule {
        role: "IR and host lowering handle-effect semantics",
        roots: &["crates/chelis-ir/src"],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    ForbiddenRule {
        role: "compiler-api evaluator handle-effect semantics",
        roots: &["crates/chelis-compiler-api/src"],
        forbidden: &["if effect == \"random\""],
    },
    ForbiddenRule {
        role: "Surf/Deep canonical effect conversion",
        roots: &["crates/chelis-surf/src"],
        forbidden: &[
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"random\"))])",
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"resource\"))])",
            "Some(\"random\") => format!",
            "Some(\"resource\") => format!",
        ],
    },
];

const RUNTIME_DTYPE_FORBIDDEN_RULES: &[ForbiddenRule] = &[
    ForbiddenRule {
        role: "runtime typed dtype dispatch",
        roots: &["crates/chelis-runtime/src"],
        forbidden: &[
            "fn tensor_elem_size(dtype: c_int)",
            "match (*t).dtype",
            "match (*tensor).dtype",
        ],
    },
    ForbiddenRule {
        role: "backend dtype macro selection",
        roots: &[
            "crates/chelis-backend-c/src",
            "crates/chelis-backend-hip/src",
            "crates/chelis-backend-metal/src",
        ],
        forbidden: &["Prim::F32 => \"CHELIS_DTYPE_F32\""],
    },
    ForbiddenRule {
        role: "published runtime/device headers and device owners",
        roots: &[
            "crates/chelis-runtime/include",
            "crates/chelis-backend-hip/runtime",
            "crates/chelis-backend-metal/runtime",
        ],
        forbidden: &[
            "return sizeof(float);",
            "case CHELIS_DTYPE_F64:",
            "case CHELIS_DTYPE_BOOL:",
        ],
    },
    ForbiddenRule {
        role: "Python FFI dtype constants",
        roots: &["crates/chelis-python/src"],
        forbidden: &["const CHELIS_DTYPE_F32: i32 = 0"],
    },
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("chelis-cli is two levels below workspace root")
        .to_path_buf()
}

fn is_scanned_source(path: &Path) -> bool {
    SourceLanguage::from_path(path).is_some()
}

fn collect_source_path(
    repository_root: &Path,
    path: &Path,
    sources: &mut Vec<ScannedSource>,
) -> Result<(), String> {
    if path.is_dir() {
        let entries = fs::read_dir(path)
            .map_err(|error| format!("cannot read source root {}: {error}", path.display()))?;
        let mut paths = entries
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("cannot enumerate source root {}: {error}", path.display()))?;
        paths.sort();
        for child in paths {
            collect_source_path(repository_root, &child, sources)?;
        }
    } else if path.is_file() && is_scanned_source(path) {
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("cannot read source {}: {error}", path.display()))?;
        let relative = path
            .strip_prefix(repository_root)
            .unwrap_or(path)
            .to_path_buf();
        sources.push(ScannedSource::new(relative, &contents));
    }
    Ok(())
}

fn scan_source_roots(root: &Path, roots: &[&str]) -> Result<Vec<ScannedSource>, String> {
    let mut sources = Vec::new();
    for relative in roots {
        let source_root = root.join(relative);
        if !source_root.exists() {
            return Err(format!("missing scanned source root `{relative}`"));
        }
        let before = sources.len();
        collect_source_path(root, &source_root, &mut sources)?;
        if sources.len() == before {
            return Err(format!(
                "scanned source root `{relative}` contains no recognized source files"
            ));
        }
    }
    sources.sort_by(|left, right| left.path.cmp(&right.path));
    sources.dedup_by(|left, right| left.path == right.path);
    Ok(sources)
}

fn is_cxx_raw_literal(spelling: &str) -> bool {
    ["R\"", "u8R\"", "uR\"", "UR\"", "LR\"", "@R\""]
        .iter()
        .any(|prefix| spelling.starts_with(prefix))
}

fn lex_c_family_tokens(source: &str) -> Vec<SourceToken> {
    c_lexical::lex_c_tokens(source)
        .into_iter()
        .map(|spelling| {
            if is_cxx_raw_literal(&spelling) {
                SourceToken::literal(SourceTokenKind::RawStringLiteral, spelling)
            } else if spelling.starts_with('"') {
                SourceToken::literal(SourceTokenKind::StringLiteral, spelling)
            } else if spelling.starts_with('\'') {
                SourceToken::literal(SourceTokenKind::CharacterLiteral, spelling)
            } else {
                SourceToken::syntax(spelling)
            }
        })
        .collect()
}

fn rust_raw_string_end(chars: &[char], start: usize) -> Option<Result<usize, String>> {
    let marker = if chars.get(start) == Some(&'r') {
        start
    } else if matches!(chars.get(start), Some('b' | 'c')) && chars.get(start + 1) == Some(&'r') {
        start + 1
    } else {
        return None;
    };

    let mut quote = marker + 1;
    while chars.get(quote) == Some(&'#') {
        quote += 1;
    }
    if chars.get(quote) != Some(&'"') {
        return None;
    }

    let hashes = quote - marker - 1;
    let mut cursor = quote + 1;
    while cursor < chars.len() {
        if chars[cursor] == '"'
            && chars
                .get(cursor + 1..cursor + 1 + hashes)
                .is_some_and(|suffix| suffix.iter().all(|character| *character == '#'))
        {
            return Some(Ok(cursor + 1 + hashes));
        }
        cursor += 1;
    }
    Some(Err("unterminated Rust raw string".to_owned()))
}

fn rust_quoted_literal_end(chars: &[char], quote: usize, delimiter: char) -> Result<usize, String> {
    let mut cursor = quote + 1;
    while cursor < chars.len() {
        if chars[cursor] == '\\' {
            cursor += 2;
        } else if chars[cursor] == delimiter {
            return Ok(cursor + 1);
        } else {
            cursor += 1;
        }
    }
    Err(format!("unterminated Rust `{delimiter}` literal"))
}

fn rust_character_literal_end(chars: &[char], quote: usize) -> Option<Result<usize, String>> {
    let first = *chars.get(quote + 1)?;
    if first == '\\' {
        return Some(rust_quoted_literal_end(chars, quote, '\''));
    }
    if chars.get(quote + 2) == Some(&'\'') {
        return Some(Ok(quote + 3));
    }
    None
}

fn lex_rust_tokens(source: &str) -> Result<Vec<SourceToken>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        if chars[index].is_whitespace() {
            index += 1;
            continue;
        }

        if chars[index] == '/' && chars.get(index + 1) == Some(&'/') {
            index += 2;
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }

        if chars[index] == '/' && chars.get(index + 1) == Some(&'*') {
            index += 2;
            let mut depth = 1usize;
            while index < chars.len() && depth > 0 {
                if chars[index] == '/' && chars.get(index + 1) == Some(&'*') {
                    depth += 1;
                    index += 2;
                } else if chars[index] == '*' && chars.get(index + 1) == Some(&'/') {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            if depth != 0 {
                return Err("unterminated Rust block comment".to_owned());
            }
            continue;
        }

        if let Some(raw_end) = rust_raw_string_end(&chars, index) {
            let end = raw_end?;
            tokens.push(SourceToken::literal(
                SourceTokenKind::RawStringLiteral,
                chars[index..end].iter().collect::<String>(),
            ));
            index = end;
            continue;
        }

        let string_quote =
            if matches!(chars[index], 'b' | 'c') && chars.get(index + 1) == Some(&'"') {
                Some(index + 1)
            } else if chars[index] == '"' {
                Some(index)
            } else {
                None
            };
        if let Some(quote) = string_quote {
            let end = rust_quoted_literal_end(&chars, quote, '"')?;
            tokens.push(SourceToken::literal(
                SourceTokenKind::StringLiteral,
                chars[index..end].iter().collect::<String>(),
            ));
            index = end;
            continue;
        }

        let character_quote = if chars[index] == 'b' && chars.get(index + 1) == Some(&'\'') {
            Some(index + 1)
        } else if chars[index] == '\'' {
            Some(index)
        } else {
            None
        };
        if let Some(quote) = character_quote
            && let Some(character_end) = rust_character_literal_end(&chars, quote)
        {
            let end = character_end?;
            tokens.push(SourceToken::literal(
                SourceTokenKind::CharacterLiteral,
                chars[index..end].iter().collect::<String>(),
            ));
            index = end;
            continue;
        }

        if chars[index].is_alphabetic() || chars[index] == '_' {
            let start = index;
            index += 1;
            while index < chars.len() && (chars[index].is_alphanumeric() || chars[index] == '_') {
                index += 1;
            }
            tokens.push(SourceToken::syntax(
                chars[start..index].iter().collect::<String>(),
            ));
            continue;
        }

        if chars[index].is_ascii_digit() {
            let start = index;
            index += 1;
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || matches!(chars[index], '_' | '.'))
            {
                index += 1;
            }
            tokens.push(SourceToken::syntax(
                chars[start..index].iter().collect::<String>(),
            ));
            continue;
        }

        tokens.push(SourceToken::syntax(chars[index].to_string()));
        index += 1;
    }

    Ok(tokens)
}

fn c_include_operand(tokens: &[SourceToken], index: usize) -> bool {
    index >= 2
        && tokens[index - 2].kind == SourceTokenKind::Syntax
        && tokens[index - 2].spelling == "#"
        && tokens[index - 1].kind == SourceTokenKind::Syntax
        && tokens[index - 1].spelling == "include"
}

fn project_evidence(language: SourceLanguage, tokens: &[SourceToken]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| match token.kind {
            SourceTokenKind::Syntax => token.spelling.clone(),
            SourceTokenKind::StringLiteral
                if language == SourceLanguage::CFamily && c_include_operand(tokens, index) =>
            {
                token.spelling.clone()
            }
            SourceTokenKind::StringLiteral
            | SourceTokenKind::CharacterLiteral
            | SourceTokenKind::RawStringLiteral => "<literal>".to_owned(),
        })
        .collect()
}

fn project_active(tokens: &[SourceToken]) -> Vec<String> {
    tokens
        .iter()
        .map(|token| match token.kind {
            SourceTokenKind::RawStringLiteral => "<raw-string>".to_owned(),
            SourceTokenKind::Syntax
            | SourceTokenKind::StringLiteral
            | SourceTokenKind::CharacterLiteral => token.spelling.clone(),
        })
        .collect()
}

fn contains_token_pattern(tokens: &[String], pattern: &[String]) -> bool {
    !pattern.is_empty()
        && tokens
            .windows(pattern.len())
            .any(|window| window == pattern)
}

fn missing_required_evidence(
    crate_name: &str,
    role: &str,
    required: &[&str],
    sources: &[ScannedSource],
) -> Vec<String> {
    required
        .iter()
        .filter(|needle| {
            !sources
                .iter()
                .any(|source| source.contains_evidence_pattern(needle))
        })
        .map(|needle| format!("{crate_name} ({role}): missing crate-level typed marker `{needle}`"))
        .collect()
}

fn forbidden_occurrences(
    vocabulary: &str,
    forbidden: &[&str],
    sources: &[ScannedSource],
) -> Vec<String> {
    let mut failures = Vec::new();
    for source in sources {
        for needle in forbidden {
            if source.contains_active_pattern(needle) {
                failures.push(format!(
                    "{}: raw {vocabulary} consumer remains: `{needle}`",
                    source.path.display()
                ));
            }
        }
    }
    failures
}

fn crate_evidence_failures(root: &Path, evidence: &[CrateEvidence]) -> Vec<String> {
    let mut failures = Vec::new();
    for consumer in evidence {
        let sources = match scan_source_roots(root, consumer.roots) {
            Ok(sources) => sources,
            Err(error) => {
                failures.push(format!(
                    "{} ({}): {error}",
                    consumer.crate_name, consumer.role
                ));
                continue;
            }
        };
        failures.extend(missing_required_evidence(
            consumer.crate_name,
            consumer.role,
            consumer.required,
            &sources,
        ));
    }
    failures
}

fn forbidden_rule_failures(root: &Path, vocabulary: &str, rules: &[ForbiddenRule]) -> Vec<String> {
    let mut failures = Vec::new();
    for rule in rules {
        let sources = match scan_source_roots(root, rule.roots) {
            Ok(sources) => sources,
            Err(error) => {
                failures.push(format!("{}: {error}", rule.role));
                continue;
            }
        };
        failures.extend(
            forbidden_occurrences(vocabulary, rule.forbidden, &sources)
                .into_iter()
                .map(|failure| format!("{}: {failure}", rule.role)),
        );
    }
    failures
}

#[test]
fn crate_level_evidence_survives_a_within_crate_file_move() {
    let temp = tempfile::tempdir().expect("temporary crate");
    let src = temp.path().join("crates/example/src");
    fs::create_dir_all(&src).expect("create example crate");
    fs::write(
        src.join("old_owner.rs"),
        "use chelis_vocab::EffectKind;\nfn decide(_: EffectKind) {}\n",
    )
    .expect("write old owner");

    let before = scan_source_roots(temp.path(), &["crates/example/src"]).expect("scan old owner");
    assert!(
        missing_required_evidence(
            "example",
            "effect consumer",
            &["chelis_vocab::EffectKind", "EffectKind"],
            &before,
        )
        .is_empty()
    );

    fs::rename(src.join("old_owner.rs"), src.join("new_owner.rs")).expect("move owner");
    let after = scan_source_roots(temp.path(), &["crates/example/src"]).expect("scan new owner");
    assert!(
        missing_required_evidence(
            "example",
            "effect consumer",
            &["chelis_vocab::EffectKind", "EffectKind"],
            &after,
        )
        .is_empty(),
        "positive evidence must belong to the crate, not an owner filename"
    );
}

#[test]
fn missing_positive_crate_evidence_fails() {
    let sources = [ScannedSource::new(
        PathBuf::from("crates/example/src/lib.rs"),
        "pub fn unrelated() {}\n",
    )];
    let failures = missing_required_evidence(
        "example",
        "runtime dtype consumer",
        &["RuntimeDType"],
        &sources,
    );
    assert_eq!(failures.len(), 1);
    assert!(failures[0].contains("RuntimeDType"), "{failures:?}");
}

#[test]
fn comments_and_strings_cannot_supply_positive_crate_evidence() {
    let temp = tempfile::tempdir().expect("temporary crate");
    let src = temp.path().join("crates/example/src");
    fs::create_dir_all(&src).expect("create example crate");
    fs::write(
        src.join("owner.rs"),
        "pub fn decide(effect: &str) -> bool { effect.is_empty() }\n",
    )
    .expect("write untyped owner");
    fs::write(
        src.join("notes.rs"),
        "// use chelis_vocab::EffectKind;\n\
         /* decode_effect_kind(list) is the intended architecture. */\n\
         const NOTE: &str = \"use chelis_vocab::EffectKind; decode_effect_kind(list)\";\n\
         const RAW_NOTE: &str = r#\"prefix \" use chelis_vocab::EffectKind; \
         decode_effect_kind(list)\"#;\n",
    )
    .expect("write non-executable markers");

    let sources =
        scan_source_roots(temp.path(), &["crates/example/src"]).expect("scan example crate");
    let failures = missing_required_evidence(
        "example",
        "effect consumer",
        &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        &sources,
    );
    assert_eq!(
        failures.len(),
        2,
        "comments and prose strings must not authenticate typed crate evidence: {failures:?}"
    );
}

#[test]
fn forbidden_raw_consumer_anywhere_in_scanned_roots_fails() {
    let temp = tempfile::tempdir().expect("temporary crate");
    let nested = temp.path().join("crates/example/src/nested");
    let include = temp.path().join("crates/example/include");
    let device = include.join("device");
    fs::create_dir_all(&nested).expect("create nested source");
    fs::create_dir_all(&device).expect("create published device headers");
    fs::write(
        nested.join("raw.rs"),
        "fn decide(effect: &str) { if effect == \"random\" {} }\n",
    )
    .expect("write raw consumer");
    fs::write(
        include.join("raw.h"),
        "static inline size_t width(void) { return sizeof(float); }\n",
    )
    .expect("write raw header consumer");
    fs::write(
        device.join("raw.cuh"),
        "__device__ int width(int dtype) {\n\
         switch (dtype) { case CHELIS_DTYPE_F64: return 8; default: return 4; }\n\
         }\n",
    )
    .expect("write raw device-header consumer");

    let sources = scan_source_roots(
        temp.path(),
        &["crates/example/src", "crates/example/include"],
    )
    .expect("scan nested source and published header");
    let failures = forbidden_occurrences(
        "closed vocabulary",
        &[
            "if effect == \"random\"",
            "return sizeof(float);",
            "case CHELIS_DTYPE_F64:",
        ],
        &sources,
    );
    assert_eq!(failures.len(), 3, "{failures:?}");
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("nested/raw.rs")),
        "{failures:?}"
    );
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("include/raw.h")),
        "{failures:?}"
    );
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("device/raw.cuh")),
        "{failures:?}"
    );
}

#[test]
fn cxx_raw_strings_do_not_supply_evidence_or_forbidden_consumers() {
    let raw_only = ScannedSource::new(
        PathBuf::from("crates/example/runtime/raw.cpp"),
        r####"
static const char *notes = R"fermat_9(
" arbitrary quote before marker text
use chelis_vocab::EffectKind
decode_effect_kind(list)
if effect == "random"
/* block opener in payload
// line opener in payload
second line
)fermat_9";
"####,
    );

    let missing = missing_required_evidence(
        "example",
        "C++ raw-string control",
        &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        &[raw_only],
    );
    assert_eq!(
        missing.len(),
        2,
        "a C++ raw-string payload must not authenticate positive evidence: {missing:?}"
    );

    let raw_only = ScannedSource::new(
        PathBuf::from("crates/example/runtime/raw.cpp"),
        r####"
static const char *notes = R"guard_42(
" quote
if effect == "random"
/* // marker text only
)guard_42";
"####,
    );
    let forbidden = forbidden_occurrences("EffectKind", &["if effect == \"random\""], &[raw_only]);
    assert!(
        forbidden.is_empty(),
        "a forbidden spelling inside a C++ raw string is not active code: {forbidden:?}"
    );
}

#[test]
fn active_forbidden_code_after_cxx_raw_string_is_detected() {
    let source = ScannedSource::new(
        PathBuf::from("crates/example/runtime/active.cpp"),
        r####"
static const char *notes = R"xY_7(
" quotes, /* block, // line, and
newlines all stay in the raw payload
)xY_7";
if effect == "random" {}
"####,
    );
    let forbidden = forbidden_occurrences("EffectKind", &["if effect == \"random\""], &[source]);
    assert_eq!(
        forbidden.len(),
        1,
        "raw-string payload syntax must not hide following active code: {forbidden:?}"
    );
}

#[test]
fn c_family_block_comments_do_not_nest() {
    let source = ScannedSource::new(
        PathBuf::from("crates/example/runtime/comment.c"),
        "/* valid C comment with an inner /* opener */\n\
         size_t width(void) { return sizeof(float); }\n",
    );
    let forbidden = forbidden_occurrences("RuntimeDType", &["return sizeof(float);"], &[source]);
    assert_eq!(
        forbidden.len(),
        1,
        "C comments end at the first `*/`; following code is active: {forbidden:?}"
    );
}

#[test]
fn rust_block_comments_remain_nested() {
    let source = ScannedSource::new(
        PathBuf::from("crates/example/src/lib.rs"),
        "/* outer /* inner */ if effect == \"random\" {} */\n\
         use chelis_vocab::EffectKind;\n\
         fn decide(list: &List, effect: &str) {\n\
             let _ = decode_effect_kind(list);\n\
             if effect == \"random\" {}\n\
         }\n",
    );
    let missing = missing_required_evidence(
        "example",
        "Rust nested-comment control",
        &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        std::slice::from_ref(&source),
    );
    assert!(missing.is_empty(), "{missing:?}");
    let forbidden = forbidden_occurrences("EffectKind", &["if effect == \"random\""], &[source]);
    assert_eq!(
        forbidden.len(),
        1,
        "the nested-comment spelling is inert and the following Rust code is active"
    );
}

#[test]
fn escaped_literals_and_include_operands_keep_language_semantics() {
    let header = ScannedSource::new(
        PathBuf::from("crates/example/include/control.hpp"),
        "#include \"chelis_runtime_dtype.h\"\n\
         const char *text = \"if effect == \\\"random\\\" /* //\";\n\
         const char quote = '\\'';\n",
    );
    let missing = missing_required_evidence(
        "example",
        "published include control",
        &["#include \"chelis_runtime_dtype.h\""],
        std::slice::from_ref(&header),
    );
    assert!(
        missing.is_empty(),
        "an active include operand remains positive evidence: {missing:?}"
    );
    let forbidden = forbidden_occurrences("EffectKind", &["if effect == \"random\""], &[header]);
    assert!(
        forbidden.is_empty(),
        "escaped ordinary literal payload is not active token structure: {forbidden:?}"
    );
}

#[test]
fn added_effect_variant_will_force_every_semantic_consumer_to_decide() {
    let root = repo_root();
    let mut failures = crate_evidence_failures(&root, EFFECT_KIND_CRATES);
    failures.extend(forbidden_rule_failures(
        &root,
        "EffectKind",
        EFFECT_KIND_FORBIDDEN_RULES,
    ));
    assert!(
        failures.is_empty(),
        "EffectKind mutation inventory is not closed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn added_runtime_dtype_will_force_runtime_and_codegen_consumers_to_decide() {
    let root = repo_root();
    let mut failures = crate_evidence_failures(&root, RUNTIME_DTYPE_CRATES);
    failures.extend(forbidden_rule_failures(
        &root,
        "RuntimeDType",
        RUNTIME_DTYPE_FORBIDDEN_RULES,
    ));
    assert!(
        failures.is_empty(),
        "RuntimeDType mutation inventory is not closed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn vocabulary_owner_has_no_dependencies() {
    let manifest = fs::read_to_string(repo_root().join("crates/chelis-vocab/Cargo.toml"))
        .expect("read chelis-vocab manifest");
    assert!(
        !manifest.contains("[dependencies]"),
        "chelis-vocab must remain a true dependency bottom; found a dependencies table"
    );
}
