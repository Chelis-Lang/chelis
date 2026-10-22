#!/usr/bin/env python3
"""Prove portable metadata, native validation, and fixed-revision CHB parity."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import tomllib

import no_std_core_support as support
from no_std_core_support import ROOT, OracleFailure, compile_rejected, consumer, portable, run

BASELINE = "7ab8af85c193cd2a29979adb5b003c9a6d19ded7"
BASELINE_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
FORBIDDEN = (
    "bincode", "chelis-shell", "chelis-syntax-host", "stacker", "psm", "miette",
    "chelis-reef", "chelis-types", "chelis-infer", "chelis-compiler-api",
    "chelis-reef-bootstrap-core", "chelis-runtime-bundle", "chelis-runtime-bundle-macro",
    "chelis-runtime", "chelis-backend-c", "chelis-backend-hip", "chelis-backend-metal",
    "tokio", "reqwest",
)
MODELS = (
    "PackageId", "ShellPackage", "ShellModule", "ShellSymbol", "SymbolKind",
    "FloatClassifyKind", "CollectionObligation", "TypeVariableRestriction", "TypeVariableDomain",
)
DOMAINS = ("active-float", "active-int", "active-numeric", "float-value", "int-value", "numeric-value", "active-set")
OBLIGATIONS = ("is_nan", "is_finite", "is_infinite", "len", "index", "append", "concat", "key_from_seed", "split_key", "split_keys", "fold_in")
MALFORMED_TYPES = (
    "empty", "syntax", "multiple", "unknown", "atom", "list", "map", "metadata-wrapper",
    "runtime-root", "runtime-child", "namespace-swap",
)
TEXT_FIELDS = ("package-name", "package-version", "module", "export", "effect", "dependency-name", "dependency-version")
TEXT_VALUES = ("empty", "space", "leading", "trailing", "tab", "newline", "unicode")
PINS = ("stable", "prerelease", "maximum", "missing-prefix", "range", "empty", "empty-version", "short", "leading-zero", "space", "double-equals", "overflow", "prerelease-zero")
VARIABLES = ("wrong-prefix", "missing-index", "leading-zero", "negative", "overflow", "gap")
FIXED_CASES = (
    "native-sample", "empty-package", "opaque-package-version", "missing-value-type", "empty-type", "canonical-gap", "old-format",
    "restrictions-absent", "restrictions-no-type", "restrictions-metadata-only", "ledger-no-type", "ledger-nonfunction",
    "ledger-malformed", "ledger-whitespace", "ledger-result-whitespace", "ledger-hidden-t", "ledger-hidden-d", "ledger-hidden-r",
    "ledger-duplicate", "ledger-descending", "ledger-same-builtin-descending", "ledger-same-builtin-sorted", "restrictions-numeric-order",
    "digest-short", "digest-long", "digest-uppercase", "digest-nonhex", "digest-unicode",
)
NATIVE_SAMPLE_HEX = (
    "4348454c434842000800000008000000040000000000000064656d6f0500000000000000312e322e3313000000000000003d312e322e332d72632e312b6275696c642e37010000000000000008000000000000004170702e44656d6f01000000000000000700000000000000666f72776172640000000001290000000000000028742d666e207b7d2028742d7072696d207b7d20663332292028742d7072696d207b7d20663332292900000000000000000000000000000000030000000000000001000000000000007a01000000000000006101000000000000007a01010000000000000003000000000000006465700000000000000000400000000000000064646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464"
)
EXACT_REJECTIONS = {
    "missing-value-type": "invalid shell envelope: value export `forward` in module `App.Demo` requires a type representation",
    "empty-type": "invalid shell envelope: type representation is not valid Deep: parse error: expected exactly one type expression, found 0 top-level expressions at byte 0",
    "canonical-gap": "invalid shell envelope: type representation type variables must be contiguous from `t0`",
    "old-format": "invalid shell envelope: payload format version 7 is unsupported; expected 8; regenerate the shell package",
}

PORTABLE_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::{string::String, vec, vec::Vec};
use chelis_shell_core::*;
use chelis_stack_services::StackServices;
pub fn metadata(stack: &impl StackServices) -> Result<ShellPackage, ShellValidationError> {
    let shell = ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId { name: "demo".into(), version: "1.2.3".into() },
        compiler: "=1.2.3".into(),
        modules: vec![ShellModule { module: "Demo".into(), exports: vec![ShellSymbol {
            name: "value".into(), kind: SymbolKind::Value,
            type_repr: Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into()),
            type_variable_restrictions: vec![TypeVariableRestriction { variable: "t0".into(), domain: TypeVariableDomain::ActiveFloat }],
            collection_obligations: vec![CollectionObligation::Len { operand: "(t-var {} t0)".into(), result: "(t-var {} t1)".into() }],
            effects: Vec::new(), has_body: false,
        }] }], dependencies: Vec::new(), archive_sha256: "a".repeat(64),
    };
    validate_shell(&shell, stack)?;
    Ok(shell)
}
pub fn ledger(stack: &impl StackServices) -> Result<(), CollectionValidationError> {
    validate_collection_obligation_ledger(&[], stack)?;
    validate_collection_obligations_for_type_repr(None, &[], stack)
}
pub fn error_text(error: &ShellValidationError) -> String {
    use alloc::string::ToString;
    error.to_string()
}
'''
CORE_CONSUMER = PORTABLE_SOURCE + r'''
extern crate std;
use chelis_syntax_host::NativeStack;
use alloc::string::ToString;
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;
fn main() {
    let mut shell = metadata(&NativeStack).unwrap();
    ledger(&NativeStack).unwrap();
    shell.archive_sha256.clear();
    assert!(matches!(validate_shell(&shell, &NativeStack), Err(ShellValidationError::ArchiveSha256)));
    shell.archive_sha256 = "a".repeat(64);
    shell.compiler = "=01.2.3".into();
    let error = validate_shell(&shell, &NativeStack).unwrap_err();
    assert!(matches!(&error, ShellValidationError::CompilerPinVersion { source } if source.to_string() == "invalid leading zero in major version number"));
    shell.compiler = "=1.2.3".into();
    shell.format_version = 7;
    assert!(matches!(validate_shell(&shell, &NativeStack), Err(ShellValidationError::UnsupportedFormatVersion { version: 7 })));
    std::println!("SHELL_CORE_CONSUMER {{\"allocator\":\"System\",\"stack\":\"NativeStack\",\"typed_errors\":3}}");
}
'''

SNAPSHOT_SOURCE = r'''
use __MODELS__::*;
use chelis_shell::{decode_shell, encode_shell, SHELL_MAGIC};
use chelis_syntax_host::NativeStack;
use serde_json::{json, Map, Value};
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;
const FUNCTION: &str = "(t-fn {} (t-var {} t0) (t-var {} t1))";
fn fixture() -> ShellPackage {
    ShellPackage {
        format_version: 8,
        package: PackageId { name: "demo".into(), version: "1.2.3".into() },
        compiler: "=1.2.3-rc.1+build.7".into(),
        modules: vec![ShellModule { module: "App.Demo".into(), exports: vec![ShellSymbol {
            name: "forward".into(), kind: SymbolKind::Value,
            type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".into()),
            type_variable_restrictions: vec![], collection_obligations: vec![],
            effects: vec!["z".into(), "a".into(), "z".into()], has_body: true,
        }] }],
        dependencies: vec![PackageId { name: "dep".into(), version: "".into() }],
        archive_sha256: "d".repeat(64),
    }
}
fn unchecked(shell: &ShellPackage) -> Vec<u8> {
    let mut bytes = SHELL_MAGIC.to_vec();
    bytes.extend(8u32.to_le_bytes());
    bytes.extend(bincode::serialize(shell).unwrap());
    bytes
}
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
fn add(cases: &mut Map<String, Value>, name: impl Into<String>, shell: ShellPackage, accepted: bool) {
    let name = name.into();
    let before = shell.clone();
    let validation = validate_shell(&shell__STACK__).map_err(|e| e.to_string());
    assert_eq!(validation.is_ok(), accepted, "case {}: {:?}", name, validation);
    assert_eq!(shell, before);
    let encoded = encode_shell(&shell).map_err(|e| e.to_string());
    let decoded = decode_shell(&unchecked(&shell)).map_err(|e| e.to_string());
    assert_eq!(encoded.is_ok(), accepted);
    assert_eq!(decoded.is_ok(), accepted);
    let record = if accepted {
        let bytes = encoded.unwrap();
        assert_eq!(decoded.as_ref().unwrap(), &shell);
        assert_eq!(encode_shell(decoded.as_ref().unwrap()).unwrap(), bytes);
        json!({"accepted": true, "value": shell, "wire": hex(&bytes), "decoded": decoded.unwrap()})
    } else {
        assert_eq!(encoded.as_ref().unwrap_err(), validation.as_ref().unwrap_err());
        assert_eq!(decoded.as_ref().unwrap_err(), validation.as_ref().unwrap_err());
        json!({"accepted": false, "validate": validation.unwrap_err(), "encode": encoded.unwrap_err(), "decode": decoded.unwrap_err()})
    };
    assert!(cases.insert(name, record).is_none());
}
fn changed(cases: &mut Map<String, Value>, name: impl Into<String>, accepted: bool, mutate: impl FnOnce(&mut ShellPackage)) {
    let mut shell = fixture(); mutate(&mut shell); add(cases, name, shell, accepted);
}
fn len(operand: &str, result: &str) -> CollectionObligation {
    CollectionObligation::Len { operand: operand.into(), result: result.into() }
}
fn contract(cases: &mut Map<String, Value>, name: &str, obligations: Vec<CollectionObligation>, repr: Option<&str>, accepted: bool) {
    let ledger = validate_collection_obligation_ledger(&obligations__STACK__).map_err(|e| e.to_string());
    let relation = validate_collection_obligations_for_type_repr(repr, &obligations__STACK__).map_err(|e| e.to_string());
    changed(cases, name, accepted, |s| {
        s.modules[0].exports[0].type_repr = repr.map(Into::into);
        s.modules[0].exports[0].collection_obligations = obligations;
    });
    cases.get_mut(name).unwrap().as_object_mut().unwrap().insert("ledger".into(), json!(ledger));
    cases.get_mut(name).unwrap().as_object_mut().unwrap().insert("relation".into(), json!(relation));
}
fn main() {
    let mut cases = Map::new();
    add(&mut cases, "native-sample", fixture(), true);
    changed(&mut cases, "empty-package", true, |s| { s.modules.clear(); s.dependencies.clear(); });
    changed(&mut cases, "opaque-package-version", true, |s| { s.package.name = "../package".into(); s.package.version = "not-semver".into(); s.dependencies[0].version = "unparsed version text".into(); });
    changed(&mut cases, "missing-value-type", false, |s| s.modules[0].exports[0].type_repr = None);
    changed(&mut cases, "empty-type", false, |s| s.modules[0].exports[0].type_repr = Some("".into()));
    changed(&mut cases, "canonical-gap", false, |s| s.modules[0].exports[0].type_repr = Some("(t-var {} t1)".into()));
    changed(&mut cases, "old-format", false, |s| s.format_version = 7);
    for (name, kind) in [("value", SymbolKind::Value), ("type", SymbolKind::Type), ("macro", SymbolKind::Macro), ("dim", SymbolKind::Dim)] {
        for body in [false, true] { for present in [false, true] {
            changed(&mut cases, format!("symbol/{name}/{body}/{present}"), present || kind != SymbolKind::Value, |s| {
                let symbol = &mut s.modules[0].exports[0]; symbol.kind = kind; symbol.has_body = body;
                if !present { symbol.type_repr = None; }
            });
        } }
    }
    for (name, domain) in [
        ("active-float", TypeVariableDomain::ActiveFloat), ("active-int", TypeVariableDomain::ActiveInt),
        ("active-numeric", TypeVariableDomain::ActiveNumeric), ("float-value", TypeVariableDomain::FloatValue),
        ("int-value", TypeVariableDomain::IntValue), ("numeric-value", TypeVariableDomain::NumericValue),
        ("active-set", TypeVariableDomain::ActiveSet(vec!["f32".into(), "f64".into()])),
    ] {
        changed(&mut cases, format!("domain/{name}"), true, |s| {
            let symbol = &mut s.modules[0].exports[0]; symbol.type_repr = Some(FUNCTION.into());
            symbol.type_variable_restrictions = vec![TypeVariableRestriction { variable: "t0".into(), domain }];
        });
    }
    for (name, pin, accepted) in [
        ("stable", "=1.2.3", true), ("prerelease", "=1.2.3-rc.1+build.7", true),
        ("maximum", "=18446744073709551615.0.0", true), ("missing-prefix", "1.2.3", false),
        ("range", "^1.2.3", false), ("empty", "", false), ("empty-version", "=", false),
        ("short", "=1.2", false), ("leading-zero", "=01.2.3", false), ("space", "=1.2.3 ", false),
        ("double-equals", "==1.2.3", false), ("overflow", "=18446744073709551616.0.0", false), ("prerelease-zero", "=1.2.3-01", false),
    ] { changed(&mut cases, format!("pin/{name}"), accepted, |s| s.compiler = pin.into()); }
    for field in ["package-name", "package-version", "module", "export", "effect", "dependency-name", "dependency-version"] {
        for (name, text) in [("empty", ""), ("space", " "), ("leading", " x"), ("trailing", "x "), ("tab", "\tx"), ("newline", "x\n"), ("unicode", "\u{2003}x")] {
            changed(&mut cases, format!("text/{field}/{name}"), field == "dependency-version" && text.is_empty(), |s| {
                let target = match field { "package-name" => &mut s.package.name, "package-version" => &mut s.package.version,
                    "module" => &mut s.modules[0].module, "export" => &mut s.modules[0].exports[0].name,
                    "effect" => &mut s.modules[0].exports[0].effects[0], "dependency-name" => &mut s.dependencies[0].name,
                    _ => &mut s.dependencies[0].version };
                *target = text.into();
            });
        }
    }
    for (name, value) in [("short", "a".repeat(63)), ("long", "a".repeat(65)), ("uppercase", "A".repeat(64)), ("nonhex", "g".repeat(64)), ("unicode", "é".repeat(32))] {
        changed(&mut cases, format!("digest-{name}"), false, |s| s.archive_sha256 = value);
    }
    for version in [0u32, 3, 4, 5, 6, 7, 9, u32::MAX] {
        changed(&mut cases, format!("payload-version/{version}"), false, |s| s.format_version = version);
    }
    for collection in ["modules", "exports", "dependencies", "restrictions"] { for duplicate in [false, true] {
        changed(&mut cases, format!("order/{collection}/{duplicate}"), false, |s| match collection {
            "modules" => { let mut other = s.modules[0].clone(); if !duplicate { other.module = "A".into(); } s.modules.push(other); },
            "exports" => { let mut other = s.modules[0].exports[0].clone(); if !duplicate { other.name = "a".into(); } s.modules[0].exports.push(other); },
            "dependencies" => { let mut other = s.dependencies[0].clone(); if !duplicate { other.name = "a".into(); } s.dependencies.push(other); },
            _ => { let symbol = &mut s.modules[0].exports[0]; symbol.type_repr = Some(FUNCTION.into());
                symbol.type_variable_restrictions = ["t1", if duplicate { "t1" } else { "t0" }].into_iter().map(|v| TypeVariableRestriction { variable: v.into(), domain: TypeVariableDomain::ActiveFloat }).collect(); },
        });
    } }
    for (namespace, tag, prefix) in [("type", "t-var", "t"), ("dimension", "d-var", "d"), ("rank", "d-rank", "r")] {
        for (name, variable) in [("wrong-prefix", "x0".to_string()), ("missing-index", prefix.into()), ("leading-zero", format!("{prefix}00")), ("negative", format!("{prefix}-1")), ("overflow", format!("{prefix}4294967296")), ("gap", format!("{prefix}1"))] {
            changed(&mut cases, format!("variable/{namespace}/{name}"), false, |s| {
                let repr = match namespace { "type" => format!("({tag} {{}} {variable})"), "dimension" => format!("(t-tensor {{}} (d-var {{}} {variable}) (d-rank {{}} r0) (t-prim {{}} f32))"), _ => format!("(t-tensor {{}} (d-var {{}} d0) (d-rank {{}} {variable}) (t-prim {{}} f32))") };
                s.modules[0].exports[0].type_repr = Some(repr);
            });
        }
    }
    for (name, variable) in [("wrong-prefix", "x0"), ("missing-index", "t"), ("leading-zero", "t00"), ("negative", "t-1"), ("overflow", "t4294967296")] {
        changed(&mut cases, format!("restriction/{name}"), false, |s| {
            s.modules[0].exports[0].type_repr = Some(FUNCTION.into());
            s.modules[0].exports[0].type_variable_restrictions = vec![TypeVariableRestriction { variable: variable.into(), domain: TypeVariableDomain::ActiveFloat }];
        });
    }
    for (name, repr, kind) in [("restrictions-absent", Some("(t-prim {} f32)"), SymbolKind::Value), ("restrictions-no-type", None, SymbolKind::Type), ("restrictions-metadata-only", Some("(t-prim {type: (t-var {} t0)} f32)"), SymbolKind::Value)] {
        changed(&mut cases, name, false, |s| { let symbol = &mut s.modules[0].exports[0]; symbol.kind = kind; symbol.type_repr = repr.map(Into::into); symbol.type_variable_restrictions = vec![TypeVariableRestriction { variable: "t0".into(), domain: TypeVariableDomain::ActiveFloat }]; });
    }
    changed(&mut cases, "restrictions-numeric-order", true, |s| {
        let symbol = &mut s.modules[0].exports[0];
        symbol.type_repr = Some((0..=10).rev().fold("(t-var {} t0)".to_string(), |tail, i| format!("(t-fn {{}} (t-var {{}} t{i}) {tail})")));
        symbol.type_variable_restrictions = (0..=10).map(|i| TypeVariableRestriction { variable: format!("t{i}"), domain: TypeVariableDomain::ActiveInt }).collect();
    });
    for (name, repr) in [
        ("empty", ""), ("syntax", "(t-unit {}"), ("multiple", "(t-unit {}) (t-unit {})"), ("unknown", "(not-a-type {} f32)"),
        ("atom", "f32"), ("list", "()"), ("map", "{type: (t-prim {} f32)}"), ("metadata-wrapper", "^{:doc \"legacy\"} (t-prim {} f32)"),
        ("runtime-root", "(lit {type: (t-var {} t0)} 1)"), ("runtime-child", "(t-fn {} (lit {type: (t-var {} t0)} 1) (t-var {} t0))"),
        ("namespace-swap", "(t-tensor {} (t-var {} t0) (d-rank {} r0))"),
    ] { for body in [false, true] { for restricted in [false, true] {
        changed(&mut cases, format!("deep/{name}/{body}/{restricted}"), false, |s| { let symbol = &mut s.modules[0].exports[0]; symbol.type_repr = Some(repr.into()); symbol.has_body = body; if restricted { symbol.type_variable_restrictions.push(TypeVariableRestriction { variable: "t0".into(), domain: TypeVariableDomain::ActiveFloat }); } });
    } } }
    let t0 = "(t-var {} t0)".to_string(); let t1 = "(t-var {} t1)".to_string();
    let all = vec![
        CollectionObligation::FloatClassify { kind: FloatClassifyKind::Nan, operand: t0.clone(), result: t1.clone() },
        CollectionObligation::FloatClassify { kind: FloatClassifyKind::Finite, operand: t0.clone(), result: t1.clone() },
        CollectionObligation::FloatClassify { kind: FloatClassifyKind::Infinite, operand: t0.clone(), result: t1.clone() },
        len(&t0, &t1), CollectionObligation::Index { list: t0.clone(), index: t0.clone(), result: t1.clone() },
        CollectionObligation::Append { list: t0.clone(), value: t1.clone(), result: t0.clone() },
        CollectionObligation::Concat { lhs: t0.clone(), rhs: t0.clone(), result: t1.clone() },
        CollectionObligation::KeyFromSeed { operand: t0.clone(), result: t1.clone() }, CollectionObligation::SplitKey { operand: t0.clone(), result: t1.clone() },
        CollectionObligation::SplitKeys { operand: t0.clone(), count: t0.clone(), result: t1.clone() },
        CollectionObligation::FoldIn { operand: t0.clone(), index: t0.clone(), result: t1.clone() },
    ];
    for obligation in all { let name = format!("obligation/{}", obligation.builtin()); contract(&mut cases, &name, vec![obligation], Some(FUNCTION), true); }
    for (name, obligations, repr) in [
        ("ledger-no-type", vec![len(&t0, &t1)], None), ("ledger-nonfunction", vec![len(&t0, &t1)], Some("(t-var {} t0)")),
        ("ledger-malformed", vec![len("List[t0]", &t1)], Some(FUNCTION)),
        ("ledger-whitespace", vec![len("(t-var   {} t0)", &t1)], Some(FUNCTION)),
        ("ledger-result-whitespace", vec![len(&t0, "(t-var {}   t1)")], Some(FUNCTION)),
        ("ledger-hidden-t", vec![len("(t-var {} t2)", &t1)], Some(FUNCTION)),
        ("ledger-hidden-d", vec![len("(t-tensor {} (d-var {} d1) (d-rank {} r0) (t-prim {} f32))", "(t-prim {} i64)")], Some("(t-fn {} (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-prim {} f32)) (t-prim {} i64))")),
        ("ledger-hidden-r", vec![len("(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-prim {} f32))", "(t-prim {} i64)")], Some("(t-fn {} (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-prim {} f32)) (t-prim {} i64))")),
        ("ledger-duplicate", vec![len(&t0, &t1), len(&t0, &t1)], Some(FUNCTION)),
        ("ledger-descending", vec![len(&t0, &t1), CollectionObligation::Append { list: t0.clone(), value: t1.clone(), result: t0.clone() }], Some(FUNCTION)),
        ("ledger-same-builtin-descending", vec![len(&t1, &t0), len(&t0, &t1)], Some(FUNCTION)),
    ] { contract(&mut cases, name, obligations, repr, false); }
    contract(&mut cases, "ledger-same-builtin-sorted", vec![len(&t0, &t1), len(&t1, &t0)], Some(FUNCTION), true);
    let failures: Vec<(&str, fn(&mut ShellPackage))> = vec![
        ("format", |s| s.format_version = 7), ("package-name", |s| s.package.name.clear()), ("package-version", |s| s.package.version.clear()),
        ("compiler", |s| s.compiler = "bad".into()), ("digest", |s| s.archive_sha256.clear()),
        ("modules-order", |s| s.modules.push(s.modules[0].clone())), ("module-name", |s| s.modules[0].module.clear()),
        ("exports-order", |s| { let symbol = s.modules[0].exports[0].clone(); s.modules[0].exports.push(symbol); }),
        ("export-name", |s| s.modules[0].exports[0].name.clear()),
        ("ledger", |s| s.modules[0].exports[0].collection_obligations = vec![len("bad", "bad")]),
        ("type", |s| s.modules[0].exports[0].type_repr = None),
        ("restriction", |s| s.modules[0].exports[0].type_variable_restrictions.push(TypeVariableRestriction { variable: "bad".into(), domain: TypeVariableDomain::ActiveInt })),
        ("effect", |s| s.modules[0].exports[0].effects.push("".into())),
        ("dependencies-order", |s| s.dependencies.push(s.dependencies[0].clone())),
        ("dependency-name", |s| s.dependencies[0].name.clear()), ("dependency-version", |s| s.dependencies[0].version = " ".into()),
    ];
    for start in 0..failures.len() {
        changed(&mut cases, format!("precedence/{}", failures[start].0), false, |s| { for (_, mutate) in &failures[start..] { mutate(s); } });
    }
    let bytes = encode_shell(&fixture()).unwrap();
    let mut wire = Map::new();
    for end in 0..bytes.len() { wire.insert(format!("truncate/{end}"), json!(decode_shell(&bytes[..end]).unwrap_err().to_string())); }
    for (name, extra) in [("zero", vec![0]), ("ff", vec![255]), ("junk", b"junk".to_vec())] {
        let mut bad = bytes.clone(); bad.extend(extra); wire.insert(format!("trailing/{name}"), json!(decode_shell(&bad).unwrap_err().to_string()));
    }
    for version in [0u32, 3, 4, 5, 6, 7, 9, u32::MAX] { let mut bad = bytes.clone(); bad[8..12].copy_from_slice(&version.to_le_bytes()); wire.insert(format!("envelope-version/{version}"), json!(decode_shell(&bad).unwrap_err().to_string())); }
    let mut bad = bytes.clone(); bad[0] = 255; wire.insert("magic".into(), json!(decode_shell(&bad).unwrap_err().to_string()));
    let mut bad = bytes.clone(); bad[24] = 255; wire.insert("utf8".into(), json!(decode_shell(&bad).unwrap_err().to_string()));
    fn offset(bytes: &[u8], field: &[u8]) -> usize {
        let found: Vec<_> = bytes.windows(field.len()).enumerate().filter_map(|(index, value)| (value == field).then_some(index)).collect();
        assert_eq!(found.len(), 1); found[0]
    }
    let shell = fixture();
    let symbol = &shell.modules[0].exports[0];
    let symbol_bytes = bincode::serialize(symbol).unwrap();
    let start = offset(&bytes, &symbol_bytes);
    let kind = start + bincode::serialized_size(&symbol.name).unwrap() as usize;
    for (name, index, width, value) in [("symbol-tag", kind, 4, 99u32), ("option-tag", kind + 4, 1, 2), ("bool-tag", start + symbol_bytes.len() - 1, 1, 2)] {
        let mut bad = bytes.clone(); bad[index..index + width].copy_from_slice(&value.to_le_bytes()[..width]);
        wire.insert(name.into(), json!(decode_shell(&bad).unwrap_err().to_string()));
    }
    for (name, field) in [("package-name", bincode::serialize(&shell.package.name).unwrap()), ("modules", bincode::serialize(&shell.modules).unwrap())] {
        let index = offset(&bytes, &field);
        for length in [0u64, 1024, u64::MAX] {
            let mut bad = bytes.clone(); bad[index..index + 8].copy_from_slice(&length.to_le_bytes());
            wire.insert(format!("length/{name}/{length}"), json!(decode_shell(&bad).unwrap_err().to_string()));
        }
    }
    let mut tagged = fixture();
    let symbol = &mut tagged.modules[0].exports[0];
    symbol.type_repr = Some(FUNCTION.into());
    symbol.type_variable_restrictions.push(TypeVariableRestriction { variable: "t0".into(), domain: TypeVariableDomain::ActiveFloat });
    symbol.collection_obligations.push(CollectionObligation::FloatClassify { kind: FloatClassifyKind::Nan, operand: t0, result: t1 });
    let tagged_bytes = encode_shell(&tagged).unwrap();
    let symbol = &tagged.modules[0].exports[0];
    let restriction = &symbol.type_variable_restrictions[0];
    let domain = offset(&tagged_bytes, &bincode::serialize(restriction).unwrap()) + bincode::serialized_size(&restriction.variable).unwrap() as usize;
    let relation = offset(&tagged_bytes, &bincode::serialize(&symbol.collection_obligations[0]).unwrap());
    for (name, index) in [("domain-tag", domain), ("relation-tag", relation), ("classification-tag", relation + 4)] {
        let mut bad = tagged_bytes.clone(); bad[index..index + 4].copy_from_slice(&99u32.to_le_bytes());
        wire.insert(name.into(), json!(decode_shell(&bad).unwrap_err().to_string()));
    }
    use bincode::Options;
    let mut variable = SHELL_MAGIC.to_vec(); variable.extend(8u32.to_le_bytes());
    variable.extend(bincode::DefaultOptions::new().with_varint_encoding().serialize(&fixture()).unwrap());
    wire.insert("varint".into(), json!(decode_shell(&variable).unwrap_err().to_string()));
    let runtime = chelis_std_bundle::CHELIS_STD_SHELL;
    let archive = chelis_std_bundle::CHELIS_STD_ARCHIVE;
    use sha2::{Digest, Sha256};
    let embedded = decode_shell(runtime).unwrap();
    assert_eq!(embedded.archive_sha256, format!("{:x}", Sha256::digest(archive)));
    assert_eq!(encode_shell(&embedded).unwrap(), runtime);
    println!("SHELL_SNAPSHOT {}", json!({"cases": cases, "wire": wire, "bundle": {"shell_sha256": format!("{:x}", Sha256::digest(runtime)), "archive_sha256": format!("{:x}", Sha256::digest(archive)), "shell": embedded}}));
}
'''


def case_inventory() -> set[str]:
    names = set(FIXED_CASES)
    names.update(f"symbol/{kind}/{body}/{present}" for kind in ("value", "type", "macro", "dim") for body in ("false", "true") for present in ("false", "true"))
    names.update(f"domain/{name}" for name in DOMAINS)
    names.update(f"obligation/{name}" for name in OBLIGATIONS)
    names.update(f"pin/{name}" for name in PINS)
    names.update(f"text/{field}/{name}" for field in TEXT_FIELDS for name in TEXT_VALUES)
    names.update(f"payload-version/{version}" for version in (0, 3, 4, 5, 6, 7, 9, 4294967295))
    names.update(f"order/{collection}/{duplicate}" for collection in ("modules", "exports", "dependencies", "restrictions") for duplicate in ("false", "true"))
    names.update(f"variable/{namespace}/{name}" for namespace in ("type", "dimension", "rank") for name in VARIABLES)
    names.update(f"restriction/{name}" for name in VARIABLES[:-1])
    names.update(f"deep/{name}/{body}/{restricted}" for name in MALFORMED_TYPES for body in ("false", "true") for restricted in ("false", "true"))
    names.update(f"precedence/{name}" for name in ("format", "package-name", "package-version", "compiler", "digest", "modules-order", "module-name", "exports-order", "export-name", "ledger", "type", "restriction", "effect", "dependencies-order", "dependency-name", "dependency-version"))
    return names


def dependencies(workspace: Path, *names: str) -> str:
    return "".join(f'{name} = {{ path = {json.dumps(str(workspace / "crates" / name))}, default-features = false }}\n' for name in names)


def parse_record(stdout: str, prefix: str) -> dict:
    records = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise OracleFailure(f"missing or repeated structured evidence: {prefix.strip()}")
    try:
        record = json.loads(records[0])
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid structured evidence: {prefix.strip()}") from error
    if not isinstance(record, dict):
        raise OracleFailure("structured evidence must be an object")
    return record


def validate_metadata_evidence(value: object) -> None:
    def fields(model: object, names: tuple[str, ...]) -> dict:
        if not isinstance(model, dict) or set(model) != set(names):
            raise OracleFailure("incomplete metadata model evidence")
        return model

    def sequence(items: object) -> list:
        if not isinstance(items, list):
            raise OracleFailure("invalid metadata collection evidence")
        return items

    def package(model: object) -> None:
        model = fields(model, ("name", "version"))
        if any(not isinstance(item, str) for item in model.values()):
            raise OracleFailure("invalid package identity evidence")

    shell = fields(value, ("format_version", "package", "compiler", "modules", "dependencies", "archive_sha256"))
    if type(shell["format_version"]) is not int or any(not isinstance(shell[key], str) for key in ("compiler", "archive_sha256")):
        raise OracleFailure("invalid metadata envelope evidence")
    package(shell["package"])
    for dependency in sequence(shell["dependencies"]):
        package(dependency)
    obligation_fields = {
        "float_classify": ("kind", "operand", "result"),
        "len": ("operand", "result"),
        "index": ("list", "index", "result"),
        "append": ("list", "value", "result"),
        "concat": ("lhs", "rhs", "result"),
        "key_from_seed": ("operand", "result"),
        "split_key": ("operand", "result"),
        "split_keys": ("operand", "count", "result"),
        "fold_in": ("operand", "index", "result"),
    }
    for module in sequence(shell["modules"]):
        module = fields(module, ("module", "exports"))
        if not isinstance(module["module"], str):
            raise OracleFailure("invalid module name evidence")
        for symbol in sequence(module["exports"]):
            symbol = fields(symbol, ("name", "kind", "type_repr", "type_variable_restrictions", "collection_obligations", "effects", "has_body"))
            if not isinstance(symbol["name"], str) or symbol["kind"] not in ("Value", "Type", "Macro", "Dim") or type(symbol["has_body"]) is not bool:
                raise OracleFailure("invalid symbol identity evidence")
            if symbol["type_repr"] is not None and not isinstance(symbol["type_repr"], str):
                raise OracleFailure("invalid type representation evidence")
            if any(not isinstance(effect, str) for effect in sequence(symbol["effects"])):
                raise OracleFailure("invalid effect evidence")
            for restriction in sequence(symbol["type_variable_restrictions"]):
                restriction = fields(restriction, ("variable", "domain"))
                if not isinstance(restriction["variable"], str):
                    raise OracleFailure("invalid restriction identity evidence")
                domain = restriction["domain"]
                if isinstance(domain, dict):
                    domain = fields(domain, ("active_set",))
                    if any(not isinstance(dtype, str) for dtype in sequence(domain["active_set"])):
                        raise OracleFailure("invalid dtype-set evidence")
                elif domain not in ("active_float", "active_int", "active_numeric", "float_value", "int_value", "numeric_value"):
                    raise OracleFailure("invalid restriction domain evidence")
            for obligation in sequence(symbol["collection_obligations"]):
                if not isinstance(obligation, dict) or len(obligation) != 1:
                    raise OracleFailure("incomplete collection contract evidence")
                kind, carried = next(iter(obligation.items()))
                if kind not in obligation_fields:
                    raise OracleFailure("invalid collection contract kind evidence")
                carried = fields(carried, obligation_fields[kind])
                if any(not isinstance(item, str) for item in carried.values()):
                    raise OracleFailure("invalid collection contract type evidence")
                if kind == "float_classify" and carried["kind"] not in ("nan", "finite", "infinite"):
                    raise OracleFailure("invalid float classification evidence")


def validate_snapshot(record: dict) -> None:
    if set(record) != {"cases", "wire", "bundle"}:
        raise OracleFailure("incomplete metadata snapshot fields")
    cases = record["cases"]
    if not isinstance(cases, dict) or set(cases) != case_inventory():
        raise OracleFailure("missing or unexpected mandatory metadata case")
    for name, case in cases.items():
        if not isinstance(case, dict) or type(case.get("accepted")) is not bool:
            raise OracleFailure(f"invalid case evidence: {name}")
        fields = {"accepted", "value", "wire", "decoded"} if case["accepted"] else {"accepted", "validate", "encode", "decode"}
        if name.startswith("ledger-") or name.startswith("obligation/"):
            fields |= {"ledger", "relation"}
        if set(case) != fields:
            raise OracleFailure(f"incomplete case fields: {name}")
        if "ledger" in fields:
            for field in ("ledger", "relation"):
                result = case[field]
                if not isinstance(result, dict) or set(result) not in ({"Ok"}, {"Err"}):
                    raise OracleFailure(f"invalid collection result evidence: {name}/{field}")
                if "Ok" in result and result["Ok"] is not None:
                    raise OracleFailure(f"invalid collection success evidence: {name}/{field}")
                if "Err" in result and (not isinstance(result["Err"], str) or not result["Err"]):
                    raise OracleFailure(f"invalid collection error evidence: {name}/{field}")
        if case["accepted"]:
            validate_metadata_evidence(case["value"])
            validate_metadata_evidence(case["decoded"])
            if case["value"] != case["decoded"] or not isinstance(case["wire"], str) or not case["wire"]:
                raise OracleFailure(f"incomplete round-trip evidence: {name}")
            try:
                bytes.fromhex(case["wire"])
            except ValueError as error:
                raise OracleFailure(f"invalid wire evidence: {name}") from error
        elif any(not isinstance(case[field], str) or not case[field] for field in ("validate", "encode", "decode")) or len({case[field] for field in ("validate", "encode", "decode")}) != 1:
            raise OracleFailure(f"incomplete rejection evidence: {name}")
    sample = cases["native-sample"]
    if sample.get("wire") != NATIVE_SAMPLE_HEX:
        raise OracleFailure("native baseline sample bytes differ")
    for name, expected in EXACT_REJECTIONS.items():
        if cases[name].get("validate") != expected:
            raise OracleFailure(f"native baseline rejection differs: {name}")
    wire = record["wire"]
    expected_wire = {f"truncate/{end}" for end in range(len(bytes.fromhex(NATIVE_SAMPLE_HEX)))}
    expected_wire |= {"trailing/zero", "trailing/ff", "trailing/junk", "magic", "utf8"}
    expected_wire |= {f"envelope-version/{v}" for v in (0, 3, 4, 5, 6, 7, 9, 4294967295)}
    expected_wire |= {"symbol-tag", "option-tag", "bool-tag", "domain-tag", "relation-tag", "classification-tag", "varint"}
    expected_wire |= {f"length/{field}/{length}" for field in ("package-name", "modules") for length in (0, 1024, 18446744073709551615)}
    if not isinstance(wire, dict) or set(wire) != expected_wire or any(not isinstance(v, str) or not v for v in wire.values()):
        raise OracleFailure("incomplete malformed-wire evidence")
    bundle = record["bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {"shell_sha256", "archive_sha256", "shell"}:
        raise OracleFailure("incomplete embedded bundle evidence")
    for field in ("shell_sha256", "archive_sha256"):
        digest = bundle[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise OracleFailure("invalid embedded digest evidence")
    validate_metadata_evidence(bundle["shell"])
    if not isinstance(bundle["shell"], dict) or bundle["shell"].get("archive_sha256") != bundle["archive_sha256"]:
        raise OracleFailure("embedded archive identity differs")


def compare_snapshots(baseline: dict, candidate: dict) -> None:
    validate_snapshot(baseline)
    validate_snapshot(candidate)
    if baseline != candidate:
        raise OracleFailure("metadata values, bytes, diagnostics, order, or embedded identity differ from the immutable base")


def checkout_baseline(directory: Path) -> Path:
    directory = directory.resolve()
    run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(directory)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=directory)
    if present.returncode != 0:
        shallow = run(("git", "rev-parse", "--is-shallow-repository"), cwd=ROOT)
        if shallow.stdout.strip() != "true":
            raise OracleFailure("immutable baseline commit is absent from a complete clone")
        run(("git", "fetch", "--quiet", "--no-tags", "--depth=1", BASELINE_REPOSITORY, BASELINE), cwd=directory)
    run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=directory)
    actual = run(("git", "rev-parse", "HEAD"), cwd=directory)
    if actual.stdout.strip() != BASELINE:
        raise OracleFailure("baseline revision differs from the required commit")
    for path in ("crates/chelis-shell/src/lib.rs", "crates/chelis-shell/Cargo.toml", "Cargo.lock", "packages/chelis-std/reef.toml"):
        if not (directory / path).is_file():
            raise OracleFailure(f"missing original baseline evidence: {path}")
    return directory


def snapshot(workspace: Path, *, candidate: bool) -> dict:
    environment = None
    if not candidate:
        environment = {
            **support._environment(),
            "CARGO_TARGET_DIR": str(workspace.resolve() / "target"),
        }
    source = SNAPSHOT_SOURCE.replace("__MODELS__", "chelis_shell_core" if candidate else "chelis_shell")
    source = source.replace("__STACK__", ", &NativeStack" if candidate else "")
    extra = dependencies(workspace, "chelis-syntax-host", "chelis-std-bundle")
    if candidate:
        extra += dependencies(workspace, "chelis-shell-core")
    result = consumer(workspace / "crates/chelis-shell", source,
                      dependencies=extra + 'serde_json = "1"\nsha2 = "0.10"\nbincode = "1"\n',
                      env=environment)
    return parse_record(result.stdout, "SHELL_SNAPSHOT ")


def target_graphs() -> None:
    for defaults in (False, True):
        names = portable("chelis-shell-core", PORTABLE_SOURCE, default_features=defaults,
                         dependencies=dependencies(ROOT, "chelis-stack-services"), forbidden=FORBIDDEN)
        if not {"chelis-shell-core", "chelis-deep", "chelis-stack-services", "serde", "semver"}.issubset(names):
            raise OracleFailure("portable graph omits a required implementation dependency")
        result = consumer("chelis-shell-core", CORE_CONSUMER, default_features=defaults,
                          dependencies=dependencies(ROOT, "chelis-stack-services", "chelis-syntax-host"))
        if parse_record(result.stdout, "SHELL_CORE_CONSUMER ") != {"allocator": "System", "stack": "NativeStack", "typed_errors": 3}:
            raise OracleFailure("incomplete native consumer evidence")


def negative_controls() -> None:
    compile_rejected("chelis-shell-core", '#![no_std]\npub fn read() { let _ = std::fs::read("x"); }\n', expected=("E0433", "std"))
    compile_rejected("chelis-shell-core", 'pub fn check(shell: &chelis_shell_core::ShellPackage) { let _ = chelis_shell_core::validate_shell(shell); }\n', expected="E0061")
    for name in (*MODELS, "SHELL_FORMAT_VERSION", "validate_shell", "validate_collection_obligation_ledger", "validate_collection_obligations_for_type_repr"):
        compile_rejected("chelis-shell", f'use chelis_shell::{name};\n', expected=name)
    with support._package("chelis-shell-core", "#![no_std]\npub use chelis_shell_core::ShellPackage;\n", executable=False, features=(), default_features=False,
                          dependencies=dependencies(ROOT, "chelis-syntax-host")) as directory:
        result = run(("cargo", "metadata", "--format-version", "1", "--filter-platform", support.TARGET), cwd=directory)
        try:
            support._normal_packages(json.loads(result.stdout), FORBIDDEN)
        except OracleFailure as error:
            if "forbidden target dependency:" not in str(error) or "chelis-syntax-host" not in str(error):
                raise OracleFailure("host dependency control failed for another reason") from error
        else:
            raise OracleFailure("host dependency control accepted native stack services")


def native_suites() -> None:
    for package, binary, names in MANDATORY_SUITES:
        run(("cargo", "nextest", "run", "--locked", "-p", package, "--test", binary, "--ignore-default-filter"), required_tests=names)
    run(("cargo", "nextest", "run", "--locked", "-p", "chelis-shell", "--lib", "--ignore-default-filter"),
        required_tests=("tests::shell_roundtrips", "tests::shell_decode_rejects_every_truncation", "tests::shell_validation_rejects_malformed_type_representations_for_every_scheme_shape"))
    run(("cargo", "nextest", "run", "--locked", "-p", "chelis-std-bundle", "--lib", "--ignore-default-filter",
         "-E", "test(=tests::archive_self_consistency) | test(=tests::embedded_shell_preserves_std_test_active_float_scheme)"),
        required_tests=("tests::archive_self_consistency", "tests::embedded_shell_preserves_std_test_active_float_scheme"))


MANDATORY_SUITES = (
    ("chelis-shell-core", "metadata_validation", (
        "full_metadata_accepts_all_symbol_and_domain_variants",
        "format_versions_reject_predecessors_and_future_values_before_other_fields",
        "text_fields_reject_empty_and_edge_whitespace_with_typed_errors",
        "compiler_pin_accepts_semver_release_prerelease_build_and_u64_boundaries",
        "compiler_pin_distinguishes_missing_prefix_from_invalid_semver",
        "archive_digest_rejects_wrong_length_case_and_non_hex_bytes",
        "modules_exports_and_dependencies_reject_duplicates_and_descending_order",
        "sorted_metadata_accepts_empty_collections_and_non_semver_package_versions",
        "value_signatures_are_required_for_both_body_states",
        "non_value_exports_accept_absent_types_but_reject_invalid_present_types",
        "envelope_errors_precede_modules_and_dependencies",
        "collection_order_precedes_invalid_member_text",
    )),
    ("chelis-shell-core", "type_boundaries", (
        "deep_types_reject_invalid_syntax_roots_and_nested_carriers_in_every_scheme_shape",
        "canonical_variable_spelling_rejects_each_namespace_and_u32_overflow",
        "canonical_variables_reject_gaps_in_each_independent_namespace",
        "canonical_variable_sets_accept_repetition_reverse_occurrence_and_independent_namespaces",
        "restriction_indices_use_numeric_order_instead_of_lexical_order",
        "restrictions_reject_duplicate_indices_and_noncanonical_identities",
        "restrictions_reject_absent_types_and_detached_or_metadata_only_variables",
        "type_errors_precede_restrictions_and_restrictions_precede_effects",
        "canonical_u32_max_indices_are_valid_identities_but_not_contiguous_schemes",
        "type_namespace_gaps_precede_dimension_gaps_and_rank_gaps",
    )),
    ("chelis-shell-core", "collection_boundaries", (
        "all_collection_variants_preserve_builtin_names_carried_types_and_valid_membership",
        "collection_variants_preserve_serde_tags_and_reject_unknown_fields",
        "empty_collection_ledgers_do_not_require_or_parse_a_function_type",
        "collection_contracts_require_a_valid_function_root",
        "every_collection_field_rejects_malformed_and_noncanonical_deep",
        "collection_ledgers_reject_duplicates_reverse_builtin_order_and_reverse_type_order",
        "every_collection_field_rejects_hidden_type_dimension_and_rank_variables",
        "collection_carried_variables_allow_sparse_subsets_but_reject_noncanonical_spelling",
        "collection_membership_ignores_variables_that_occur_only_in_metadata",
        "collection_contracts_accept_declared_shape_variables_in_every_carried_field",
        "collection_ledger_errors_precede_function_and_variable_errors",
        "shell_collection_errors_keep_export_context_and_precede_missing_value_types",
    )),
    ("chelis-shell", "host_behavior", (
        "host_codec_matches_native_chb8_baseline_bytes", "host_codec_rejects_every_native_sample_truncation",
        "host_codec_rejects_trailing_data", "host_codec_rejects_malformed_magic_and_predecessor_payloads",
        "host_codec_rejects_envelope_versions_before_payload_decode", "host_codec_rejects_payload_versions_on_encode_and_decode",
        "host_codec_preserves_current_relation_and_restriction_fields",
        "host_codec_enforces_encode_decode_validation_parity", "host_codec_rejects_invalid_utf8",
        "host_codec_rejects_invalid_option_enum_and_bool_tags", "host_codec_rejects_invalid_relation_and_domain_tags",
        "host_codec_rejects_invalid_string_and_collection_lengths", "host_filesystem_round_trip_retains_native_baseline_bytes",
        "host_filesystem_validation_failure_preserves_existing_file", "host_filesystem_invalid_metadata_does_not_create_file",
        "host_filesystem_reports_read_and_write_io_errors", "host_filesystem_io_failure_preserves_existing_parent_file",
        "host_filesystem_retains_decode_errors_for_corrupt_files",
    )),
    ("chelis-reef", "artifact_validation", (
        "valid_artifact_pair_verifies_without_writing", "artifact_pair_rejects_mismatched_archive_sha",
        "artifact_pair_rejects_trailing_and_truncated_chb", "artifact_pair_rejects_decodable_mutation_in_unconsumed_metadata",
        "artifact_pair_rejects_noncanonical_metadata_order", "installation_rejects_trailing_chb_before_registry_mutation",
    )),
    ("chelis-cli", "reef_verify_artifact", (
        "verify_artifact_accepts_valid_pair_in_human_and_machine_modes", "verify_artifact_machine_failure_is_json_only_and_obeys_invariant",
        "verify_artifact_rejects_truncation_mutation_and_archive_mismatch",
    )),
)


HOST_TRUST_SOURCE = r'''
use chelis_shell::{encode_shell, read_shell, write_shell};
use chelis_shell_core::{PackageId, ShellPackage, SHELL_FORMAT_VERSION};
use chelis_reef::{install_validated_artifact_pair, verify_artifact_pair};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;
fn tree(root: &Path) -> Vec<(String, Vec<u8>)> {
    fn visit(base: &Path, path: &Path, rows: &mut Vec<(String, Vec<u8>)>) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() { visit(base, &path, rows); }
            else { rows.push((path.strip_prefix(base).unwrap().to_string_lossy().into(), fs::read(path).unwrap())); }
        }
    }
    let mut rows = Vec::new(); visit(root, root, &mut rows); rows.sort(); rows
}
fn cli_install(monorepo: &Path, registry: &Path, expected: &str) {
    let before = tree(registry);
    let output = Command::new(__CLI__).args(["reef", "install", "demo=1.2.3", "--from-monorepo"])
        .arg(monorepo).env("CHELIS_REEF_HOME", registry).env("CHELIS_STYLE_GATE_DISABLE", "1")
        .output().expect("execute real CLI");
    assert!(!output.status.success());
    let diagnostic = String::from_utf8(output.stderr).unwrap();
    assert!(diagnostic.contains(expected), "{diagnostic}");
    assert_eq!(tree(registry), before);
}
fn main() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let monorepo = root.join("monorepo");
    let package = monorepo.join("packages/demo");
    let dist = package.join("dist"); fs::create_dir_all(&dist).unwrap();
    fs::write(package.join("reef.toml"), "[package]\nname = \"demo\"\nversion = \"1.2.3\"\ncompiler = \"=__COMPILER_VERSION__\"\nmodule_prefix = \"Demo\"\n").unwrap();
    let archive = dist.join("demo-1.2.3.tar.zst");
    let chb = dist.join("demo-1.2.3.chb");
    let registry = root.join("registry"); fs::create_dir(&registry).unwrap();
    fs::write(registry.join("sentinel"), b"prior registry state").unwrap();
    let before = tree(&registry);
    let archive_bytes = b"real archive trust input";
    fs::write(&archive, archive_bytes).unwrap();
    let digest = format!("{:x}", Sha256::digest(archive_bytes));
    let mut shell = ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId { name: "demo".into(), version: "1.2.3".into() },
        compiler: "=__COMPILER_VERSION__".into(), modules: vec![], dependencies: vec![], archive_sha256: digest.clone(),
    };
    write_shell(&chb, &shell).unwrap();
    assert_eq!(read_shell(&chb).unwrap(), shell);
    assert_eq!(fs::read(&chb).unwrap(), encode_shell(&shell).unwrap());
    assert_eq!(verify_artifact_pair(&archive, &chb).unwrap().package, shell.package);
    assert_eq!(tree(&registry), before);
    let output = Command::new(__CLI__).args(["reef", "verify-artifact", "--archive"]).arg(&archive)
        .arg("--shell").arg(&chb).arg("--json").output().unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["valid"], true);
    shell.archive_sha256 = "a".repeat(64); write_shell(&chb, &shell).unwrap();
    assert!(verify_artifact_pair(&archive, &chb).unwrap_err().contains("archive_sha256"));
    let error = install_validated_artifact_pair(&archive, &chb, "demo", "1.2.3", &registry, None).unwrap_err();
    assert!(error.contains("archive_sha256")); assert_eq!(tree(&registry), before);
    cli_install(&monorepo, &registry, "archive_sha256");
    shell.archive_sha256 = digest;
    for (name, version) in [("other", "1.2.3"), ("demo", "9.9.9")] {
        shell.package = PackageId { name: name.into(), version: version.into() };
        write_shell(&chb, &shell).unwrap();
        let error = install_validated_artifact_pair(&archive, &chb, "demo", "1.2.3", &registry, None).unwrap_err();
        assert!(error.contains("but was requested as `demo-1.2.3`")); assert_eq!(tree(&registry), before);
        cli_install(&monorepo, &registry, "but was requested as `demo-1.2.3`");
    }
    shell.package = PackageId { name: "demo".into(), version: "1.2.3".into() };
    let mut corrupt = encode_shell(&shell).unwrap(); corrupt.extend(b"junk"); fs::write(&chb, corrupt).unwrap();
    assert!(install_validated_artifact_pair(&archive, &chb, "demo", "1.2.3", &registry, None).unwrap_err().contains("decode"));
    cli_install(&monorepo, &registry, "decode");
    assert_eq!(tree(&registry), before);
    println!("SHELL_HOST_TRUST {{\"reef\":true,\"cli\":true,\"archive_digest\":true,\"requested_name\":true,\"requested_version\":true,\"registry_unchanged\":true,\"trailing_rejected\":true}}");
}
'''


def host_trust() -> None:
    run(("cargo", "build", "--locked", "-p", "chelis-cli", "--bin", "chelis"))
    target = Path(support._environment()["CARGO_TARGET_DIR"])
    if not target.is_absolute():
        target = ROOT / target
    binary = target / "debug" / "chelis"
    if not binary.is_file():
        raise OracleFailure("real CLI executable is absent")
    with (ROOT / "Cargo.toml").open("rb") as manifest:
        compiler_version = tomllib.load(manifest)["workspace"]["package"]["version"]
    source = HOST_TRUST_SOURCE.replace("__CLI__", json.dumps(str(binary))).replace("__COMPILER_VERSION__", compiler_version)
    result = consumer("chelis-shell-core", source,
                      dependencies=dependencies(ROOT, "chelis-shell", "chelis-reef") + 'tempfile = "3"\nsha2 = "0.10"\nserde_json = "1"\n')
    expected = dict.fromkeys(("reef", "cli", "archive_digest", "requested_name", "requested_version", "registry_unchanged", "trailing_rejected"), True)
    if parse_record(result.stdout, "SHELL_HOST_TRUST ") != expected:
        raise OracleFailure("incomplete host trust evidence")


def main() -> int:
    target_graphs()
    negative_controls()
    with tempfile.TemporaryDirectory(prefix="chelis-shell-baseline-") as temporary:
        baseline = checkout_baseline(Path(temporary) / "baseline")
        previous = snapshot(baseline, candidate=False)
        current = snapshot(ROOT, candidate=True)
        compare_snapshots(previous, current)
        print("SHELL_COMPARISON " + json.dumps({"baseline": BASELINE, "metadata_cases": len(case_inventory()), "wire_cases": len(current["wire"]), "bundle": current["bundle"]["shell_sha256"]}, sort_keys=True), flush=True)
    native_suites()
    host_trust()
    from test_shell_metadata_core_oracle import MANDATORY_TESTS
    result = run((sys.executable, "scripts/test_shell_metadata_core_oracle.py"))
    if parse_record(result.stdout, "SHELL_ORACLE_TESTS ") != {"passed": sorted(MANDATORY_TESTS), "skipped": 0}:
        raise OracleFailure("incomplete oracle failure-control test evidence")
    print("Target library compilation is not device execution.", flush=True)
    print("shell metadata core oracle: PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleFailure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"shell metadata core oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
