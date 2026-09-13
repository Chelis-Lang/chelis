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
    code: String,
    evidence: String,
}

impl ScannedSource {
    fn new(path: PathBuf, contents: &str) -> Self {
        let code = source_without_comments(contents);
        let evidence = source_tokens_for_evidence(&code);
        Self {
            path,
            code,
            evidence,
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
            "Some(\"random\") => format!(\"with seed",
            "Some(\"resource\") => format!(\"with device",
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
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("rs")
            | Some("h")
            | Some("hh")
            | Some("hpp")
            | Some("hxx")
            | Some("cuh")
            | Some("c")
            | Some("cc")
            | Some("cpp")
            | Some("cu")
            | Some("m")
            | Some("mm")
    )
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

fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    let first = *bytes.get(start + 1)?;
    if first == b'\\' {
        let mut index = start + 3;
        while index < bytes.len() && bytes[index] != b'\'' && bytes[index] != b'\n' {
            index += 1;
        }
        return (bytes.get(index) == Some(&b'\'')).then_some(index + 1);
    }

    let width = match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return None,
    };
    (bytes.get(start + 1 + width) == Some(&b'\'')).then_some(start + 2 + width)
}

fn rust_raw_string_span(bytes: &[u8], start: usize) -> Option<(usize, usize, usize)> {
    let mut marker = start;
    if matches!(bytes.get(marker), Some(b'b' | b'c')) {
        marker += 1;
    }
    if bytes.get(marker) != Some(&b'r') {
        return None;
    }

    let mut quote = marker + 1;
    while bytes.get(quote) == Some(&b'#') {
        quote += 1;
    }
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }

    let hashes = quote - marker - 1;
    let content_start = quote + 1;
    let mut closing_quote = content_start;
    while closing_quote < bytes.len() {
        if bytes[closing_quote] == b'"'
            && bytes
                .get(closing_quote + 1..closing_quote + 1 + hashes)
                .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
        {
            return Some((content_start, closing_quote, closing_quote + 1 + hashes));
        }
        closing_quote += 1;
    }

    Some((content_start, bytes.len(), bytes.len()))
}

fn source_without_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut code = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if let Some((_, _, end)) = rust_raw_string_span(bytes, index) {
            code.extend_from_slice(&bytes[index..end]);
            index = end;
            continue;
        }

        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            code.extend_from_slice(b"  ");
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                code.push(if bytes[index] == b'\r' { b'\r' } else { b' ' });
                index += 1;
            }
            continue;
        }

        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            code.extend_from_slice(b"  ");
            index += 2;
            let mut depth = 1usize;
            while index < bytes.len() && depth > 0 {
                if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
                    code.extend_from_slice(b"  ");
                    index += 2;
                    depth += 1;
                } else if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
                    code.extend_from_slice(b"  ");
                    index += 2;
                    depth -= 1;
                } else {
                    code.push(match bytes[index] {
                        b'\n' => b'\n',
                        b'\r' => b'\r',
                        _ => b' ',
                    });
                    index += 1;
                }
            }
            continue;
        }

        if bytes[index] == b'"' {
            let quote = bytes[index];
            code.push(quote);
            index += 1;
            while index < bytes.len() {
                code.push(bytes[index]);
                if bytes[index] == b'\\' {
                    index += 1;
                    if index < bytes.len() {
                        code.push(bytes[index]);
                    }
                } else if bytes[index] == quote {
                    index += 1;
                    break;
                }
                index += 1;
            }
            continue;
        }

        if bytes[index] == b'\''
            && let Some(end) = char_literal_end(bytes, index)
        {
            code.extend_from_slice(&bytes[index..end]);
            index = end;
            continue;
        }

        code.push(bytes[index]);
        index += 1;
    }

    String::from_utf8(code).expect("comment removal preserves UTF-8")
}

fn quote_is_include_operand(code: &[u8], quote: usize) -> bool {
    let line_start = code[..quote]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |newline| newline + 1);
    let compact = code[line_start..quote]
        .iter()
        .copied()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    compact == b"#include"
}

fn source_tokens_for_evidence(code: &str) -> String {
    let bytes = code.as_bytes();
    let mut evidence = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if let Some((content_start, content_end, end)) = rust_raw_string_span(bytes, index) {
            evidence.extend_from_slice(&bytes[index..content_start]);
            for byte in &bytes[content_start..content_end] {
                evidence.push(match byte {
                    b'\n' => b'\n',
                    b'\r' => b'\r',
                    _ => b' ',
                });
            }
            evidence.extend_from_slice(&bytes[content_end..end]);
            index = end;
            continue;
        }

        if bytes[index] == b'"' {
            let preserve = quote_is_include_operand(bytes, index);
            evidence.push(b'"');
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    evidence.push(if preserve { b'\\' } else { b' ' });
                    index += 1;
                    if index < bytes.len() {
                        evidence.push(if preserve { bytes[index] } else { b' ' });
                        index += 1;
                    }
                } else if bytes[index] == b'"' {
                    evidence.push(b'"');
                    index += 1;
                    break;
                } else {
                    evidence.push(match bytes[index] {
                        b'\n' => b'\n',
                        b'\r' => b'\r',
                        byte if preserve => byte,
                        _ => b' ',
                    });
                    index += 1;
                }
            }
            continue;
        }

        if bytes[index] == b'\''
            && let Some(end) = char_literal_end(bytes, index)
        {
            evidence.push(b'\'');
            for byte in &bytes[index + 1..end - 1] {
                evidence.push(match byte {
                    b'\n' => b'\n',
                    b'\r' => b'\r',
                    _ => b' ',
                });
            }
            evidence.push(b'\'');
            index = end;
            continue;
        }

        evidence.push(bytes[index]);
        index += 1;
    }

    String::from_utf8(evidence).expect("token projection preserves UTF-8")
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
                .any(|source| source.evidence.contains(**needle))
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
            if source.code.contains(needle) {
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
