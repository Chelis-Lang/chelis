//! The capacity census tripwire: chelis#729 / `spec/design/dtype_semantics.md`
//! §C6 deliverable 1 (PR #950), the pre-Phase-1 guard of the surface-growth
//! ratchet.
//!
//! It enumerates the public numeric surface from the ARTIFACTS (never a
//! hand-maintained list): the published runtime headers' transitive
//! `#include "..."` closure, and the desugared Deep AST of every
//! `packages/chelis-std/src` source. The inventory is diffed against the
//! checked-in baseline `spec/design/capacity_census.json`.
//!
//! Sanctioned actions when this test fails (also printed in the failure
//! message, which is the contract - a context-poor agent reads only that):
//!
//! 1. UNFLAGGED surface addition: regenerate the baseline with
//!    `CHELIS_CAPACITY_CENSUS_WRITE=1 cargo test -p chelis-cli --test
//!    capacity_census_tripwire`, then replace the generated
//!    `"citation": "TODO"` with an OPEN chelis issue reference. The test
//!    fails while any TODO remains, so regeneration alone can never
//!    self-bless.
//! 2. FLAGGED capacity seam: NO citation path exists (PR #950 red team
//!    P1-1) - the grandfathered 2026-07-30 seam set is frozen by exact
//!    citation string and count. Redesign onto the tagged carrier, remove
//!    the surface, or obtain a `maintainer-override(...)` citation, which
//!    only a human reviewer adds (`AGENTS.md` §Numeric Surface
//!    Discipline).
//! 3. NEW numeric callable: add its exact `SemanticRegistration` to a newly
//!    authored normative `[05-OP-N]` atom; a maintainer capacity override
//!    does not waive this independent semantics obligation.
//! 4. A removed row is an ABI removal and is 0.19 payload by default
//!    (`spec/design/remediation_roadmap.md` anti-churn invariant 7).
//!
//! This file and the baseline are guard artifacts: editing either to make a
//! change pass is never the fix. Deferred legs (wire-schema numeric fields,
//! binding-side raw-dtype parameters) remain typed, fixed manifest entries;
//! relabeling JSON cannot claim an enumerator or mutation oracle exists.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use chelis_deep::tag::DeepTag;
use chelis_deep::{Atom, Expr, List};
use serde::{Deserialize, Serialize};

const BASELINE_REL: &str = "spec/design/capacity_census.json";
const INCLUDE_DIR_REL: &str = "crates/chelis-runtime/include";
/// Roots of the published header surface: `chelis_runtime.h`'s transitive
/// closure is the tarball surface; `chelis_blas.h` and `chelis_math.h` are
/// emitted as `#include`s by the C backend, so generated programs compile
/// against them too.
const HEADER_ROOTS: &[&str] = &["chelis_runtime.h", "chelis_blas.h", "chelis_math.h"];
const STD_SRC_REL: &str = "packages/chelis-std/src";
const CONTROLLING_SPEC_REL: &str = "spec/05-risc-primitives.md";
const NUMERIC_PRIMS: &[&str] = &[
    "f64", "f32", "f16", "bf16", "int8", "int16", "int32", "int64",
];

/// The exact citation carried by the grandfathered 2026-07-30 capacity
/// seams. A FLAGGED row (float-carrier / raw-dtype-int) has NO
/// issue-citation path (PR #950 red team P1-1: an open-issue path would
/// make the known-red set monotonically growable): its citation must be
/// this string (the frozen pre-ratchet set) or a
/// `maintainer-override(...)` marker, which only a human reviewer adds -
/// the baseline file is review-routed.
const GRANDFATHER_SEAM_CITATION: &str = "baseline-2026-07-30 pre-ratchet seam; \
unwinds with chelis#893 (the Repr-keyed payload seal) and the 0.19 storage break";

/// The frozen seam set may only SHRINK. Copying the grandfather citation
/// onto a new flagged row trips the count lock AND the identity lock below.
const GRANDFATHER_SEAM_COUNT: usize = 21;

/// The plain-baseline citation for non-seam pre-ratchet rows.
const GRANDFATHER_PLAIN_CITATION: &str =
    "baseline-2026-07-30 pre-ratchet surface (chelis#729 C6 initial census)";

/// The exact (id) identities of the frozen 2026-07-30 seam set. Living in
/// THIS file rather than the regeneratable baseline is the point (PR #950
/// re-red-team P1: a count-only freeze permits removing one seam and
/// relocating its citation onto a brand-new one).
// GRANDFATHER_SEAM_IDS_BEGIN
const GRANDFATHER_SEAM_IDS: &[&str] = &[
    "chelis_runtime.h: chelis_string chelis_string_from_f32(float value);",
    "chelis_runtime.h: chelis_string chelis_string_from_f64(double value);",
    "chelis_runtime.h: chelis_tensor *chelis_alloc(int ndim, const int *shape, int dtype);",
    "chelis_runtime.h: chelis_tensor *chelis_alloc_view(int ndim, const int *shape, int dtype, float *data);",
    "chelis_runtime.h: chelis_tensor *chelis_scalar_tensor_from_f32(float value);",
    "chelis_runtime.h: chelis_tensor *chelis_scalar_tensor_from_f64(double value);",
    "chelis_runtime.h: chelis_tensor *chelis_tensor_from_value_list_typed(const chelis_list *list, int dst_dtype);",
    "chelis_runtime.h: chelis_value chelis_value_from_f64(double value);",
    "chelis_runtime.h: double chelis_tensor_to_f64(const chelis_tensor *t);",
    "chelis_runtime.h: double chelis_value_as_f64(chelis_value value);",
    "chelis_runtime.h: int chelis_dtype_size(int dtype);",
    "chelis_runtime.h: int chelis_format_shortest(double value, int dtype, char *buf, size_t cap);",
    "chelis_runtime.h: void chelis_bf16_buffer_to_f32(const uint16_t *src, float *dst, int64_t n);",
    "chelis_runtime.h: void chelis_f16_buffer_to_f32(const uint16_t *src, float *dst, int64_t n);",
    "chelis_runtime.h: void chelis_f32_buffer_to_bf16(const float *src, uint16_t *dst, int64_t n);",
    "chelis_runtime.h: void chelis_f32_buffer_to_f16(const float *src, uint16_t *dst, int64_t n);",
    "chelis_runtime.h: void chelis_fill_f32(chelis_tensor *t, float val);",
    "chelis_runtime.h: void chelis_fill_f64(chelis_tensor *t, double val);",
    "chelis_runtime.h: typedef struct { _Bool is_some; double value; } chelis_option_f64",
    "chelis_runtime.h: typedef struct { chelis_value_tag tag; union { int64_t i64; double f64; _Bool boolean; chelis_string string; chelis_tensor *tensor; chelis_list *list; chelis_tuple *tuple; chelis_dict *dict; chelis_adt *adt; } as; } chelis_value",
    "chelis_runtime.h: typedef struct { float *data; int shape[8]; int strides[8]; int ndim; int dtype; int size; int owns_data; } chelis_tensor",
];
// GRANDFATHER_SEAM_IDS_END

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Row {
    kind: String,
    id: String,
    #[serde(default)]
    flags: Vec<String>,
    citation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CoveredLeg {
    leg: String,
    artifact: String,
    enumerator: String,
    command: String,
    expected_success: String,
    mutations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DeferredLeg {
    leg: String,
    owner: String,
    artifact: String,
    enumerator: String,
    command: String,
    expected_success: String,
    mutations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CoverageManifest {
    covered: Vec<CoveredLeg>,
    deferred: Vec<DeferredLeg>,
}

fn coverage_manifest() -> CoverageManifest {
    CoverageManifest {
        covered: vec![
            CoveredLeg {
                leg: "header-export".to_string(),
                artifact: "crates/chelis-runtime/include/*.h published closure".to_string(),
                enumerator: "preprocessed_headers -> header_rows".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "reviewer_preprocessor_capacity_seam_is_visible".to_string(),
                    "c_identity_is_token_canonical_across_preprocessor_whitespace".to_string(),
                ],
            },
            CoveredLeg {
                leg: "header-struct".to_string(),
                artifact: "crates/chelis-runtime/include/*.h published closure".to_string(),
                enumerator: "preprocessed_headers -> header_rows".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec!["planted_struct_layout_is_inventoried".to_string()],
            },
            CoveredLeg {
                leg: "std-adt-numeric".to_string(),
                artifact: "packages/chelis-std/src/**/*.ch desugared Deep AST".to_string(),
                enumerator: "stdlib_rows -> scan_deftypes + scan_exported_numeric_defs".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec![
                    "std_adt_identity_changes_when_same_dtype_variant_changes".to_string(),
                ],
            },
            CoveredLeg {
                leg: "std-def-numeric".to_string(),
                artifact: "packages/chelis-std/src/**/*.ch desugared Deep AST".to_string(),
                enumerator: "stdlib_rows -> scan_deftypes + scan_exported_numeric_defs".to_string(),
                command: "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
                    .to_string(),
                expected_success: "capacity_census_matches_public_surface passes".to_string(),
                mutations: vec!["exported_public_numeric_stdlib_def_is_enumerated".to_string()],
            },
        ],
        deferred: vec![
            DeferredLeg {
                leg: "wire-schema-numeric-fields".to_string(),
                owner: "chelis#729 Phase 1 entry hard edge".to_string(),
                artifact: "crates/chelis-compiler-api/src/schema.rs public serde/JsonSchema graph"
                    .to_string(),
                enumerator: "PLANNED: typed public wire-schema numeric-field enumerator"
                    .to_string(),
                command:
                    "PLANNED: cargo nextest run -p chelis-compiler-api --test capacity_census_wire"
                        .to_string(),
                expected_success: "PLANNED: exact schema rows match a reviewed baseline"
                    .to_string(),
                mutations: vec![
                    "PLANNED: add/remove public f64 serde/JsonSchema field".to_string(),
                    "CURRENT DEFERRED PROBE: ReviewerWireNumericProbe leaves this census unchanged"
                        .to_string(),
                ],
            },
            DeferredLeg {
                leg: "binding-raw-dtype-params".to_string(),
                owner: "chelis#729 Phase 1 entry hard edge".to_string(),
                artifact: "crates/chelis-python/src/lib.rs registered PyO3 callables".to_string(),
                enumerator: "PLANNED: rustdoc-JSON PyO3 callable-signature enumerator".to_string(),
                command:
                    "PLANNED: cargo nextest run -p chelis-python --test capacity_census_bindings"
                        .to_string(),
                expected_success: "PLANNED: exact binding rows match a reviewed baseline"
                    .to_string(),
                mutations: vec![
                    "PLANNED: add/remove registered #[pyfunction] dtype: i32 parameter".to_string(),
                    "CURRENT DEFERRED PROBE: reviewer_raw_dtype_probe leaves this census unchanged"
                        .to_string(),
                ],
            },
        ],
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    version: u32,
    legs: CoverageManifest,
    rows: Vec<Row>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticRegistration {
    callable: &'static str,
    atom: &'static str,
}

/// Empty until spec/05 acquires its first `[05-OP-N]` atom. Existing
/// numeric rows are the frozen pre-ratchet baseline; every future callable
/// requires an exact entry here and the controlling atom in the same change.
const SEMANTIC_REGISTRATIONS: &[SemanticRegistration] = &[];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

// ---------------------------------------------------------------------------
// Leg A: published header exports and struct layouts
// ---------------------------------------------------------------------------

/// Preprocess every root and merge the per-header outputs. The include
/// closure is followed by the real preprocessor, so an export added to a
/// transitively-included header - or hidden behind a macro - is visible.
fn preprocessed_headers(include_dir: &Path, roots: &[&str]) -> BTreeMap<String, String> {
    assert_context_invariant_headers(include_dir, roots);
    let mut per_file: BTreeMap<String, String> = BTreeMap::new();
    for root in roots {
        for (name, text) in preprocess_root(include_dir, root) {
            if let Some(previous) = per_file.get(&name) {
                let previous_rows = header_rows_local(&name, previous);
                let current_rows = header_rows_local(&name, &text);
                assert_eq!(
                    previous_rows,
                    current_rows,
                    "{}CONTEXT-VARYING PUBLIC ABI in `{name}`: two published \
                     roots preprocess it to different exported declarations. \
                     Published ABI must be context-invariant; move the \
                     conditional behind a static implementation detail.{}",
                    teaching_header(),
                    teaching_footer()
                );
            } else {
                per_file.insert(name, text);
            }
        }
    }
    per_file
}

fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    rest.split('"').next()
}

fn header_source_closure(include_dir: &Path, roots: &[&str]) -> BTreeMap<String, String> {
    let mut pending: Vec<String> = roots.iter().map(|root| (*root).to_string()).collect();
    let mut sources = BTreeMap::new();
    while let Some(name) = pending.pop() {
        if sources.contains_key(&name) {
            continue;
        }
        let path = include_dir.join(&name);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for line in source.lines() {
            if let Some(included) = quoted_include(line)
                && include_dir.join(included).is_file()
            {
                pending.push(included.to_string());
            }
        }
        sources.insert(name, source);
    }
    sources
}

fn include_guard_name(source: &str) -> Option<String> {
    let directives: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('#'))
        .take(2)
        .collect();
    let guard = directives.first()?.strip_prefix("#ifndef")?.trim();
    let defined = directives.get(1)?.strip_prefix("#define")?.trim();
    (guard == defined).then(|| guard.to_string())
}

/// The supported preprocessing policy is total by construction: public ABI
/// declarations in the published local-header closure may not be conditional.
/// Platform and feature branches remain permitted inside `static` function
/// bodies and for include/macro selection that does not declare ABI.
fn assert_context_invariant_headers(include_dir: &Path, roots: &[&str]) {
    let sources = header_source_closure(include_dir, roots);
    for (name, source) in &sources {
        let guard = include_guard_name(source);
        let mut conditional_stack: Vec<bool> = Vec::new();
        let mut brace_depth = 0usize;
        let mut conditional_top_level = String::new();
        let mut conditional_macros = BTreeSet::new();
        let mut conditional_includes = BTreeSet::new();

        for line in strip_c_comments(source).lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("#ifdef")
                .or_else(|| trimmed.strip_prefix("#ifndef"))
                .or_else(|| trimmed.strip_prefix("#if"))
            {
                let is_guard = conditional_stack.is_empty()
                    && guard.as_deref().is_some_and(|g| rest.trim() == g);
                conditional_stack.push(!is_guard);
                continue;
            }
            if trimmed.starts_with("#elif") || trimmed == "#else" {
                continue;
            }
            if trimmed.starts_with("#endif") {
                conditional_stack.pop();
                continue;
            }

            let varying = conditional_stack.iter().any(|frame| *frame);
            if varying {
                if let Some(rest) = trimmed.strip_prefix("#define")
                    && let Some(macro_name) = rest.split_whitespace().next()
                {
                    conditional_macros.insert(
                        macro_name
                            .split('(')
                            .next()
                            .expect("split always has first")
                            .to_string(),
                    );
                }
                if let Some(included) = quoted_include(trimmed) {
                    conditional_includes.insert(included.to_string());
                }
            }
            let extern_wrapper = trimmed == "extern \"C\" {" || trimmed == "}";
            if varying && brace_depth == 0 && !trimmed.starts_with('#') {
                conditional_top_level.push_str(line);
                conditional_top_level.push('\n');
            }
            if !extern_wrapper {
                for c in line.chars() {
                    if c == '{' {
                        brace_depth += 1;
                    } else if c == '}' {
                        brace_depth = brace_depth.saturating_sub(1);
                    }
                }
            }
        }

        let conditional_rows = header_rows_local(name, &conditional_top_level);
        let conditional_typedef = conditional_top_level
            .split(';')
            .any(|statement| normalize_ws(statement).starts_with("typedef "));
        let raw_rows = header_rows_local(name, source);
        let macro_dependent_rows: Vec<&Row> = raw_rows
            .iter()
            .filter(|row| {
                let tokens: BTreeSet<&str> = row
                    .id
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .filter(|token| !token.is_empty())
                    .collect();
                conditional_macros
                    .iter()
                    .any(|macro_name| tokens.contains(macro_name.as_str()))
            })
            .collect();
        let conditional_include_rows: Vec<Row> = conditional_includes
            .iter()
            .filter_map(|included| sources.get(included).map(|source| (included, source)))
            .flat_map(|(included, source)| header_rows_local(included, source))
            .collect();
        assert!(
            conditional_rows.is_empty()
                && !conditional_typedef
                && macro_dependent_rows.is_empty()
                && conditional_include_rows.is_empty(),
            "{}CONTEXT-VARYING PUBLIC ABI in `{name}`: conditional branches \
             contain exported declarations {:?}, a top-level typedef is \
             conditional ({conditional_typedef}), conditional macros reach \
             declarations {:?}, or conditional local includes expose {:?}. \
             Published ABI declarations and their type spellings must be \
             unconditional in the local header closure; conditional code is \
             permitted only behind static implementation details.{}",
            teaching_header(),
            conditional_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            macro_dependent_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            conditional_include_rows
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            teaching_footer()
        );
    }
}

fn strip_c_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            out.push(' ');
        } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tokenize the declaration subset of C used by published headers. Identity
/// is the token sequence, never a preprocessor's incidental whitespace.
fn canonical_c_tokens(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '"' || c == '\'' {
            let quote = c;
            let mut token = String::new();
            token.push(c);
            i += 1;
            while i < chars.len() {
                let next = chars[i];
                token.push(next);
                i += 1;
                if next == '\\' && i < chars.len() {
                    token.push(chars[i]);
                    i += 1;
                } else if next == quote {
                    break;
                }
            }
            tokens.push(token);
            continue;
        }
        if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
            let mut token = String::new();
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '.')
            {
                token.push(chars[i]);
                i += 1;
            }
            tokens.push(token);
            continue;
        }
        if i + 2 < chars.len() && chars[i..i + 3] == ['.', '.', '.'] {
            tokens.push("...".to_string());
            i += 3;
            continue;
        }
        if i + 1 < chars.len() {
            let pair = [c, chars[i + 1]].iter().collect::<String>();
            if matches!(
                pair.as_str(),
                "->" | "++" | "--" | "<<" | ">>" | "<=" | ">=" | "==" | "!=" | "&&" | "||"
            ) {
                tokens.push(pair);
                i += 2;
                continue;
            }
        }
        tokens.push(c.to_string());
        i += 1;
    }
    tokens.join(" ")
}

fn canonical_inventory_id(id: &str) -> String {
    if let Some((header, declaration)) = id.split_once(": ")
        && header.ends_with(".h")
    {
        return format!("{header}: {}", canonical_c_tokens(declaration));
    }
    id.to_string()
}

/// True when the citation names at least one chelis issue (`chelis#N`), so
/// the liveness gate (`scripts/capacity_census_liveness.py`) has purchase on
/// every sanctioned citation - including `maintainer-override(...)`, which
/// must name its issue per §C6.
fn cites_a_chelis_issue(citation: &str) -> bool {
    citation
        .match_indices("chelis#")
        .any(|(i, m)| citation[i + m.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// Fixed-width numeric C value types: a signature mentioning one (after
/// typedef resolution) is a numeric runtime callable and carries the
/// `numeric-op` flag, which binds NEW rows to semantic registration (the
/// PR #950 re-red-team's P1 finding: surface existence is not a semantic
/// decision). Bare `int` is deliberately absent - it is dtype-id/ndim
/// plumbing, and the raw-dtype-int seam rule handles its dangerous shape.
const NUMERIC_C_TYPES: &[&str] = &[
    "double", "float", "int64_t", "int32_t", "int16_t", "int8_t", "uint64_t", "uint32_t",
    "uint16_t", "uint8_t",
];

/// The flags that make a row a capacity SEAM (subject to the grandfather
/// freeze). `numeric-op` is classification, not a seam.
fn is_seam(flags: &[String]) -> bool {
    flags
        .iter()
        .any(|f| f == "float-carrier" || f == "raw-dtype-int")
}

/// Collect simple `typedef <target...> <name>;` aliases (no struct bodies)
/// so classification sees through spellings like
/// `typedef int chelis_dtype_id;` - the re-red-team's executed typedef
/// evasion. Struct forward typedefs resolve to their `struct X` spelling,
/// which is harmless.
fn collect_typedefs(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut map = BTreeMap::new();
    for stmt in text.split(';') {
        let stmt = normalize_ws(stmt);
        if let Some(rest) = stmt.strip_prefix("typedef ")
            && !rest.contains('{')
            && !rest.contains('(')
        {
            let mut words: Vec<String> = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .filter(|w| !w.is_empty())
                .map(|w| w.to_string())
                .collect();
            if words.len() >= 2 {
                let name = words.pop().expect("nonempty");
                map.insert(name, words);
            }
        }
    }
    map
}

/// Expand typedef aliases (transitively, depth-capped) so classification
/// operates on resolved spellings.
fn resolve_words(words: Vec<String>, typedefs: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut out = words;
    for _ in 0..8 {
        let mut changed = false;
        let mut next = Vec::with_capacity(out.len());
        for w in &out {
            if let Some(target) = typedefs.get(w) {
                next.extend(target.iter().cloned());
                changed = true;
            } else {
                next.push(w.clone());
            }
        }
        out = next;
        if !changed {
            break;
        }
    }
    out
}

/// Classification shapes the enforcement rule a row falls under; the
/// citation requirement applies to EVERY inventory change, so renaming a
/// parameter to dodge a flag dodges nothing, and typedef/macro spellings
/// are resolved before classifying.
fn classify(sig: &str, typedefs: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut flags = Vec::new();
    let words: Vec<String> = sig
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    let words = resolve_words(words, typedefs);
    if words.iter().any(|w| w == "double" || w == "float") {
        flags.push("float-carrier".to_string());
    }
    // A raw `int` (never int8_t/int32_t/uint32_t, which are exact-width
    // spellings) adjacent to an identifier mentioning dtype: the
    // `(value, int dtype)` seam shape.
    for pair in words.windows(2) {
        if pair[0] == "int" && pair[1].contains("dtype") {
            flags.push("raw-dtype-int".to_string());
            break;
        }
    }
    if words.iter().any(|w| NUMERIC_C_TYPES.contains(&w.as_str())) {
        flags.push("numeric-op".to_string());
    }
    flags
}

/// Run the REAL C preprocessor over a root header and return its output
/// attributed per header file via linemarkers, restricted to files under
/// `include_dir` (system-header content is dropped). This is the
/// compiled-artifact requirement made literal: `#define`-hidden spellings
/// arrive expanded, so the re-red-team's macro evasion is visible. A
/// missing C compiler fails LOUDLY - a skip here would be an evasion
/// channel.
fn preprocess_root(include_dir: &Path, root: &str) -> BTreeMap<String, String> {
    let out = std::process::Command::new("cc")
        .arg("-E")
        .arg("-x")
        .arg("c")
        .arg("-I")
        .arg(include_dir)
        .arg(include_dir.join(root))
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "{}the capacity census requires a C compiler (`cc`) on PATH to \
                 preprocess the published headers; none ran: {e}{}",
                teaching_header(),
                teaching_footer()
            )
        });
    assert!(
        out.status.success(),
        "cc -E failed for {root}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let dir_str = include_dir.to_string_lossy().to_string();
    let mut per_file: BTreeMap<String, String> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            // Linemarker: `# <num> "<file>" <flags...>`.
            if let Some(file) = rest.split('"').nth(1) {
                current = if file.contains(&dir_str) || file.ends_with(root) {
                    Path::new(file)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                } else {
                    None
                };
            }
            continue;
        }
        if let Some(name) = &current {
            per_file.entry(name.clone()).or_default().push_str(line);
            per_file.entry(name.clone()).or_default().push('\n');
        }
    }
    per_file
}

/// Extract exported declarations and struct layouts from one preprocessed
/// header body. `static` definitions carry no ABI and are skipped; the
/// `extern "C" {` wrapper is neutralized; preprocessor lines are dropped.
/// Planted-test convenience: classify with the typedefs found in the
/// same text (the real pipeline builds a global map across headers).
fn header_rows_local(header_name: &str, raw: &str) -> Vec<Row> {
    let typedefs = collect_typedefs(&strip_c_comments(raw));
    header_rows(header_name, raw, &typedefs)
}

fn header_rows(header_name: &str, raw: &str, typedefs: &BTreeMap<String, Vec<String>>) -> Vec<Row> {
    let text = strip_c_comments(raw);
    let text: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("extern \"C\" {", "");

    let mut rows = Vec::new();
    let mut seg = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                let head = normalize_ws(&seg);
                if head.starts_with("typedef struct") {
                    // Capture the brace-matched body plus the trailing name.
                    let mut depth = 1usize;
                    let mut body = String::new();
                    for c2 in chars.by_ref() {
                        match c2 {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        body.push(c2);
                    }
                    let mut tail = String::new();
                    for c2 in chars.by_ref() {
                        if c2 == ';' {
                            break;
                        }
                        tail.push(c2);
                    }
                    let declaration = format!(
                        "{} {{ {} }} {}",
                        head,
                        normalize_ws(&body),
                        normalize_ws(&tail)
                    );
                    let flags = classify(&declaration, typedefs);
                    rows.push(Row {
                        kind: "header-struct".to_string(),
                        id: format!("{header_name}: {}", canonical_c_tokens(&declaration)),
                        flags,
                        citation: String::new(),
                    });
                } else {
                    // A definition body (static inline etc.): no ABI export;
                    // skip to the matching close brace and drop the head.
                    let mut depth = 1usize;
                    for c2 in chars.by_ref() {
                        match c2 {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                seg.clear();
            }
            '}' => {
                // Orphaned closer from the neutralized extern "C" block.
                seg.clear();
            }
            ';' => {
                let stmt = normalize_ws(&seg);
                seg.clear();
                if stmt.is_empty() || stmt.starts_with("typedef") || stmt.starts_with("static") {
                    continue;
                }
                if stmt.contains('(') && stmt.ends_with(')') {
                    let flags = classify(&stmt, typedefs);
                    rows.push(Row {
                        kind: "header-export".to_string(),
                        id: format!("{header_name}: {}", canonical_c_tokens(&format!("{stmt};"))),
                        flags,
                        citation: String::new(),
                    });
                }
            }
            _ => seg.push(c),
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// Leg B: numeric ADT variants in the desugared stdlib AST
// ---------------------------------------------------------------------------

fn walk_ch_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            walk_ch_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "ch") {
            out.push(path);
        }
    }
}

fn collect_numeric_tprims(expr: &Expr, prims: &mut BTreeSet<String>) {
    match expr {
        Expr::List(list, _) => {
            if list.tag() == Some(DeepTag::TPrim)
                && let Some(Expr::Atom(Atom::Symbol(name), _)) = list.elements.get(2)
                && NUMERIC_PRIMS.contains(&name.as_str())
            {
                prims.insert(name.clone());
            }
            for e in &list.elements {
                collect_numeric_tprims(e, prims);
            }
        }
        Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                collect_numeric_tprims(v, prims);
            }
        }
        Expr::MetaExpr(me, _) => collect_numeric_tprims(&me.expr, prims),
        Expr::Atom(..) => {}
    }
}

fn deftype_name(list: &List) -> String {
    for e in list.elements.iter().skip(2) {
        if let Expr::Atom(Atom::Symbol(name), _) = e {
            return name.clone();
        }
    }
    "<unnamed>".to_string()
}

fn symbol(expr: &Expr) -> Option<&str> {
    if let Expr::Atom(Atom::Symbol(name), _) = expr {
        Some(name)
    } else {
        None
    }
}

fn scan_exported_numeric_defs(list: &List, file_label: &str, rows: &mut Vec<Row>) {
    if list.tag() != Some(DeepTag::Module) {
        return;
    }
    let declarations = list.elements.iter().skip(3);
    let mut exports = BTreeSet::new();
    let mut signatures: BTreeMap<String, &Expr> = BTreeMap::new();
    for declaration in declarations.clone() {
        let Expr::List(declaration, _) = declaration else {
            continue;
        };
        match declaration.tag() {
            Some(DeepTag::Export) => {
                exports.extend(
                    declaration
                        .elements
                        .iter()
                        .skip(2)
                        .filter_map(symbol)
                        .map(str::to_string),
                );
            }
            Some(DeepTag::Defsig) => {
                if let (Some(name), Some(signature)) = (
                    declaration.elements.get(2).and_then(symbol),
                    declaration.elements.get(3),
                ) {
                    signatures.insert(name.to_string(), signature);
                }
            }
            _ => {}
        }
    }
    for name in exports {
        let Some(signature) = signatures.get(&name) else {
            continue;
        };
        let mut prims = BTreeSet::new();
        collect_numeric_tprims(signature, &mut prims);
        if prims.is_empty() {
            continue;
        }
        rows.push(Row {
            kind: "std-def-numeric".to_string(),
            id: format!(
                "{file_label}::{name}: {}",
                chelis_deep::printer::print_expr_flat(signature)
            ),
            flags: vec!["numeric-op".to_string()],
            citation: String::new(),
        });
    }
}

fn scan_deftypes(exprs: &[Expr], file_label: &str, rows: &mut Vec<Row>) {
    fn walk(expr: &Expr, file_label: &str, rows: &mut Vec<Row>) {
        match expr {
            Expr::List(list, _) => {
                scan_exported_numeric_defs(list, file_label, rows);
                if list.tag() == Some(DeepTag::Deftype) {
                    let mut prims = BTreeSet::new();
                    for e in list.elements.iter().skip(2) {
                        collect_numeric_tprims(e, &mut prims);
                    }
                    if !prims.is_empty() {
                        let shape = list
                            .elements
                            .iter()
                            .skip(3)
                            .map(chelis_deep::printer::print_expr_flat)
                            .collect::<Vec<_>>()
                            .join(" ");
                        let id = format!("{file_label}::{}: {shape}", deftype_name(list),);
                        rows.push(Row {
                            kind: "std-adt-numeric".to_string(),
                            id,
                            flags: Vec::new(),
                            citation: String::new(),
                        });
                    }
                }
                for e in &list.elements {
                    walk(e, file_label, rows);
                }
            }
            Expr::Map(map, _) => {
                for (_, v) in &map.entries {
                    walk(v, file_label, rows);
                }
            }
            Expr::MetaExpr(me, _) => walk(&me.expr, file_label, rows),
            Expr::Atom(..) => {}
        }
    }
    for e in exprs {
        walk(e, file_label, rows);
    }
}

fn stdlib_rows(root: &Path) -> Vec<Row> {
    let src_dir = root.join(STD_SRC_REL);
    let mut files = Vec::new();
    walk_ch_files(&src_dir, &mut files);
    files.sort();
    let mut rows = Vec::new();
    for path in files {
        let src =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let decls = chelis_surf::parser::parse_str(&src).unwrap_or_else(|e| {
            panic!(
                "stdlib source must parse for the capacity census: {}: {e:?}",
                path.display()
            )
        });
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let label = path
            .strip_prefix(&src_dir)
            .expect("under src dir")
            .with_extension("")
            .to_string_lossy()
            .replace('\\', "/");
        scan_deftypes(&exprs, &label, &mut rows);
    }
    rows
}

// ---------------------------------------------------------------------------
// The inventory, the baseline, and the diff
// ---------------------------------------------------------------------------

fn current_inventory(root: &Path) -> Vec<Row> {
    let include_dir = root.join(INCLUDE_DIR_REL);
    let mut rows = Vec::new();
    let per_file = preprocessed_headers(&include_dir, HEADER_ROOTS);
    let mut typedefs = BTreeMap::new();
    for text in per_file.values() {
        typedefs.append(&mut collect_typedefs(text));
    }
    for (name, text) in &per_file {
        rows.extend(header_rows(name, text, &typedefs));
    }
    rows.extend(stdlib_rows(root));
    rows.sort_by(|a, b| (a.kind.as_str(), a.id.as_str()).cmp(&(b.kind.as_str(), b.id.as_str())));
    rows.dedup_by(|a, b| a.kind == b.kind && a.id == b.id);
    rows
}

fn teaching_header() -> String {
    "capacity census violation \
     (spec/design/dtype_semantics.md §C6 deliverable 1; \
     AGENTS.md §Numeric Surface Discipline; chelis#729)\n"
        .to_string()
}

fn teaching_footer() -> String {
    "\nSanctioned actions:\n\
     1. UNFLAGGED addition (no capacity shape): regenerate with \
     CHELIS_CAPACITY_CENSUS_WRITE=1 cargo test -p chelis-cli --test \
     capacity_census_tripwire, then replace the generated citation TODO with \
     an OPEN chelis issue reference. A citation invented to pass this gate \
     is the defect, not a fix.\n\
     2. FLAGGED capacity seam (float-carrier / raw-dtype-int): there is NO \
     citation path - opening a fresh issue is not authorization. Redesign \
     onto the tagged carrier, remove the surface, or obtain a \
     maintainer-override(<reason>, chelis#N) citation, which only a human \
     reviewer adds (the baseline file is review-routed).\n\
     3. NEW numeric-op callable: cite its chelis#N issue, author one exact \
     [05-OP-N] atom in spec/05, and add its exact `SemanticRegistration` \
     mapping in this file. An unrelated or nonexistent atom is not authority.\n\
     4. A removed row is an ABI removal: 0.19 payload by default per \
     remediation_roadmap.md anti-churn invariant 7.\n\
     This test and spec/design/capacity_census.json are guard artifacts; \
     editing either to make a change pass is never the fix.\n"
        .to_string()
}

fn check_against_baseline(current: &[Row], baseline: &Baseline) -> Result<(), String> {
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL))
        .expect("controlling spec/05 must be readable");
    check_against_baseline_with(current, baseline, SEMANTIC_REGISTRATIONS, &spec)
}

fn callable_identity(row: &Row) -> String {
    format!("[{}] {}", row.kind, row.id)
}

fn registration_problem(registration: SemanticRegistration, spec: &str) -> Option<String> {
    let Some(atom) = registration
        .atom
        .strip_prefix('[')
        .and_then(|atom| atom.strip_suffix(']'))
    else {
        return Some(format!(
            "malformed atom `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    };
    let parts: Vec<&str> = atom.split('-').collect();
    if parts.len() != 3
        || parts[0] != "05"
        || parts[1] != "OP"
        || parts[2].is_empty()
        || !parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return Some(format!(
            "wrong atom grammar/group `{}` (expected `[05-OP-N]`)",
            registration.atom
        ));
    }
    let definition_prefix = format!("> **{}**", registration.atom);
    if !spec
        .lines()
        .any(|line| line.trim_start().starts_with(&definition_prefix))
    {
        return Some(format!(
            "atom `{}` does not exist as a normative `> **[05-OP-N]**` \
             definition in {}",
            registration.atom, CONTROLLING_SPEC_REL
        ));
    }
    None
}

fn check_against_baseline_with(
    current: &[Row],
    baseline: &Baseline,
    registrations: &[SemanticRegistration],
    spec: &str,
) -> Result<(), String> {
    let base_map: BTreeMap<(String, String), &Row> = baseline
        .rows
        .iter()
        .map(|r| ((r.kind.clone(), r.id.clone()), r))
        .collect();
    let cur_keys: BTreeSet<(String, String)> = current
        .iter()
        .map(|r| (r.kind.clone(), r.id.clone()))
        .collect();

    let mut problems = Vec::new();
    let expected_manifest = coverage_manifest();
    if baseline.version != 2 || baseline.legs != expected_manifest {
        problems.push(format!(
            "INVALID COVERAGE MANIFEST: baseline version/legs do not equal \
             the fixed executable `coverage_manifest`; deferred \
             wire-schema-numeric-fields and binding-raw-dtype-params cannot \
             become covered without a live enumerator, command, expected \
             success, and mutation_oracle. expected={expected_manifest:?}, \
             actual(version={}, legs={:?})",
            baseline.version, baseline.legs
        ));
    }

    let mut registration_map = BTreeMap::new();
    for registration in registrations {
        if registration_map
            .insert(registration.callable, *registration)
            .is_some()
        {
            problems.push(format!(
                "DUPLICATE SEMANTIC REGISTRATION for `{}`",
                registration.callable
            ));
        }
        if let Some(problem) = registration_problem(*registration, spec) {
            problems.push(format!(
                "INVALID SEMANTIC REGISTRATION for `{}`: {problem}",
                registration.callable
            ));
        }
    }

    for row in current {
        if let Some(baseline_row) = base_map.get(&(row.kind.clone(), row.id.clone())) {
            if row.flags != baseline_row.flags {
                problems.push(format!(
                    "ENFORCEMENT METADATA CHANGED for matched row [{}] {}: \
                     baseline flags {:?}, current flags {:?}",
                    row.kind, row.id, baseline_row.flags, row.flags
                ));
            }
        } else {
            problems.push(format!(
                "NEW surface not in the census: [{}] {} (flags: {:?})",
                row.kind, row.id, row.flags
            ));
        }
    }
    for row in &baseline.rows {
        if !cur_keys.contains(&(row.kind.clone(), row.id.clone())) {
            problems.push(format!(
                "REMOVED surface still in the census: [{}] {}",
                row.kind, row.id
            ));
        }
        if row.citation.trim().is_empty() || row.citation.trim() == "TODO" {
            problems.push(format!(
                "UNCITED census row (citation is TODO/empty): [{}] {}",
                row.kind, row.id
            ));
            continue;
        }
        if is_seam(&row.flags)
            && row.citation != GRANDFATHER_SEAM_CITATION
            && !row.citation.starts_with("maintainer-override(")
        {
            problems.push(format!(
                "NEW capacity seam without a sanctioned disposition (an issue \
                 citation is NOT a path for flagged rows): [{}] {}",
                row.kind, row.id
            ));
        }
        if row.citation == GRANDFATHER_SEAM_CITATION
            && !GRANDFATHER_SEAM_IDS
                .iter()
                .any(|id| canonical_inventory_id(id) == row.id)
        {
            problems.push(format!(
                "GRANDFATHER citation on an identity outside the frozen \
                 2026-07-30 seam set (identity relocation; the set may only \
                 shrink): [{}] {}",
                row.kind, row.id
            ));
        }
        if (row.flags.iter().any(|f| f == "numeric-op") || row.kind == "std-def-numeric")
            && row.citation != GRANDFATHER_PLAIN_CITATION
            && row.citation != GRANDFATHER_SEAM_CITATION
        {
            let callable = callable_identity(row);
            match registration_map.get(callable.as_str()) {
                None => problems.push(format!(
                    "NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION: \
                     `{callable}` has no `SemanticRegistration`; citation \
                     `{}` is not a callable-to-authority mapping",
                    row.citation
                )),
                Some(registration) => {
                    if let Some(problem) = registration_problem(*registration, spec) {
                        problems.push(format!(
                            "NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION: \
                             `{callable}` maps to `{}` but {problem}",
                            registration.atom
                        ));
                    }
                }
            }
        }
        if !cites_a_chelis_issue(&row.citation) {
            problems.push(format!(
                "CITATION NAMES NO ISSUE (every sanctioned citation carries a \
                 chelis#N reference so the liveness gate has purchase; prose \
                 is not a citation): [{}] {}",
                row.kind, row.id
            ));
        }
    }
    let grandfathered = baseline
        .rows
        .iter()
        .filter(|r| !r.flags.is_empty() && r.citation == GRANDFATHER_SEAM_CITATION)
        .count();
    if grandfathered > GRANDFATHER_SEAM_COUNT {
        problems.push(format!(
            "grandfathered seam citation appears on {grandfathered} rows; the \
             frozen 2026-07-30 set is {GRANDFATHER_SEAM_COUNT} and may only \
             shrink"
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}{}\n{}",
            teaching_header(),
            problems.join("\n"),
            teaching_footer()
        ))
    }
}

fn regenerate(baseline_path: &Path, current: &[Row], old: Option<&Baseline>) {
    let old_citations: BTreeMap<(String, String), String> = old
        .map(|b| {
            b.rows
                .iter()
                .map(|r| {
                    (
                        (r.kind.clone(), canonical_inventory_id(&r.id)),
                        r.citation.clone(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let rows: Vec<Row> = current
        .iter()
        .map(|r| Row {
            kind: r.kind.clone(),
            id: r.id.clone(),
            flags: r.flags.clone(),
            citation: old_citations
                .get(&(r.kind.clone(), r.id.clone()))
                .cloned()
                .unwrap_or_else(|| "TODO".to_string()),
        })
        .collect();
    let out = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows,
    };
    let json = serde_json::to_string_pretty(&out).expect("serialize census");
    fs::write(baseline_path, json + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", baseline_path.display()));
}

// ---------------------------------------------------------------------------
// The tripwire
// ---------------------------------------------------------------------------

#[test]
fn capacity_census_matches_public_surface() {
    let root = repo_root();
    let baseline_path = root.join(BASELINE_REL);
    let current = current_inventory(&root);
    assert!(
        !current.is_empty(),
        "capacity census enumerated nothing; the enumerators are broken, \
         which would make every future addition invisible"
    );

    let old: Option<Baseline> = fs::read_to_string(&baseline_path)
        .ok()
        .map(|text| serde_json::from_str(&text).expect("parse capacity_census.json"));

    if std::env::var("CHELIS_CAPACITY_CENSUS_WRITE").as_deref() == Ok("1") {
        regenerate(&baseline_path, &current, old.as_ref());
    }

    let baseline: Baseline =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap_or_else(|e| {
            panic!(
                "{}missing baseline {}: {e}\nRun once with \
                 CHELIS_CAPACITY_CENSUS_WRITE=1 to create it, then fill the \
                 TODO citations.{}",
                teaching_header(),
                baseline_path.display(),
                teaching_footer()
            )
        }))
        .expect("parse capacity_census.json");

    if let Err(msg) = check_against_baseline(&current, &baseline) {
        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Planted-evasion unit tests (the negative parity for the guard itself)
// ---------------------------------------------------------------------------

#[test]
fn planted_dtype_int_export_is_flagged() {
    // The chelis#891 pad_sequences shape: a (value, int dtype) pair.
    let rows = header_rows_local(
        "planted.h",
        "chelis_tensor *chelis_pad_sequences(const chelis_list *sequences,\n\
         chelis_value pad_value, int pad_dtype);\n",
    );
    assert_eq!(rows.len(), 1, "planted export must be enumerated: {rows:?}");
    assert!(
        rows[0].flags.contains(&"raw-dtype-int".to_string()),
        "int pad_dtype must classify as raw-dtype-int: {rows:?}"
    );
}

#[test]
fn planted_multiline_and_float_carrier() {
    let rows = header_rows_local(
        "planted.h",
        "double chelis_read_scalar(\n    const chelis_tensor *t,\n    int index);\n",
    );
    assert_eq!(rows.len(), 1);
    assert!(rows[0].flags.contains(&"float-carrier".to_string()));
    assert!(
        !rows[0].flags.contains(&"raw-dtype-int".to_string()),
        "an int param without a dtype-ish name is inventoried but unflagged: {rows:?}"
    );
}

#[test]
fn planted_static_inline_carries_no_abi_row() {
    let rows = header_rows_local(
        "planted.h",
        "static inline float bits_to_f32(uint32_t b) { return 0.0f; }\n\
         void chelis_real_export(int x);\n",
    );
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        ["planted.h: void chelis_real_export ( int x ) ;"],
        "static inline must be skipped, the real export kept"
    );
}

#[test]
fn planted_struct_layout_is_inventoried() {
    let rows = header_rows_local(
        "planted.h",
        "typedef struct {\n  int dtype;\n  double f64_;\n} planted_value;\n",
    );
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].kind, "header-struct");
    assert!(rows[0].flags.contains(&"float-carrier".to_string()));
    assert!(rows[0].flags.contains(&"raw-dtype-int".to_string()));
}

#[test]
fn planted_deftype_with_f64_variant_is_detected() {
    // The chelis#891 JNum shape, built through the typed Deep constructors
    // (artifact-level, not text): (deftype {} Json (variant JNum (t-prim {} f64))).
    let span = chelis_deep::Span::new(0, 0);
    let tprim = Expr::node(
        DeepTag::TPrim,
        Default::default(),
        vec![Expr::Atom(Atom::Symbol("f64".to_string()), span)],
        span,
    );
    let deftype = Expr::node(
        DeepTag::Deftype,
        Default::default(),
        vec![Expr::Atom(Atom::Symbol("Json".to_string()), span), tprim],
        span,
    );
    let mut rows = Vec::new();
    scan_deftypes(&[deftype], "planted", &mut rows);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].id, "planted::Json: (t-prim {} f64)");
}

#[test]
fn todo_citation_fails_with_teaching_message() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void x(void);".to_string(),
        flags: vec![],
        citation: "TODO".to_string(),
    };
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let current = vec![row];
    let err = check_against_baseline(&current, &baseline).unwrap_err();
    assert!(err.contains("UNCITED"), "{err}");
    assert!(
        err.contains("spec/design/dtype_semantics.md §C6")
            && err.contains("AGENTS.md §Numeric Surface Discipline")
            && err.contains("A citation invented to pass this gate is the defect"),
        "the failure message must teach the rule and the sanctioned actions: {err}"
    );
}

#[test]
fn new_and_removed_rows_fail() {
    let cited = |id: &str| Row {
        kind: "header-export".to_string(),
        id: id.to_string(),
        flags: vec![],
        citation: "baseline-2026-07-30".to_string(),
    };
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![cited("a.h: void old(void);")],
    };
    let current = vec![cited("a.h: void brand_new(void);")];
    let err = check_against_baseline(&current, &baseline).unwrap_err();
    assert!(err.contains("NEW surface"), "{err}");
    assert!(err.contains("REMOVED surface"), "{err}");
    assert!(err.contains("anti-churn invariant 7"), "{err}");
}

fn flagged_row(id: &str, citation: &str) -> Row {
    Row {
        kind: "header-export".to_string(),
        id: id.to_string(),
        flags: vec!["raw-dtype-int".to_string()],
        citation: citation.to_string(),
    }
}

/// PR #950 red team P1-1: opening a fresh issue and citing it must NOT
/// bless a new capacity seam - flagged rows have no issue-citation path.
#[test]
fn new_flagged_seam_cannot_be_cited_with_an_issue() {
    let row = flagged_row(
        "planted.h: void f(chelis_value v, int pad_dtype);",
        "chelis#123456",
    );
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("NEW capacity seam") && err.contains("NOT a path for flagged rows"),
        "an issue citation must not bless a flagged row: {err}"
    );
    assert!(
        err.contains("opening a fresh issue is not authorization"),
        "the teaching message must state the P1-1 rule: {err}"
    );
}

/// Copying the grandfather citation string onto an extra flagged row
/// trips the count lock: the frozen seam set may only shrink.
#[test]
fn grandfather_citation_cannot_be_copied_onto_new_rows() {
    let rows: Vec<Row> = (0..=GRANDFATHER_SEAM_COUNT)
        .map(|i| {
            flagged_row(
                &format!("planted.h: void f{i}(int x_dtype);"),
                GRANDFATHER_SEAM_CITATION,
            )
        })
        .collect();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: rows.clone(),
    };
    let err = check_against_baseline(&rows, &baseline).unwrap_err();
    assert!(
        err.contains("may only shrink"),
        "the grandfather count lock must trip: {err}"
    );
}

/// The one human exception: a maintainer-override citation passes, and
/// its issue references stay under the liveness gate.
#[test]
fn maintainer_override_is_the_human_exception() {
    let row = flagged_row(
        "planted.h: void staged(int out_dtype);",
        "maintainer-override(FFI staging for chelis#893, chelis#893)",
    );
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    assert!(check_against_baseline(&[row], &baseline).is_ok());
}

#[test]
fn maintainer_override_does_not_waive_numeric_semantic_registration() {
    let mut row =
        header_rows_local("planted.h", "double staged(double value, int out_dtype);").remove(0);
    row.citation = "maintainer-override(FFI staging, chelis#893)".to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION"),
        "capacity disposition and callable semantics are independent: {err}"
    );
}

/// Prose is not a citation: every sanctioned citation names a chelis
/// issue so the liveness gate has purchase (PR #950 §C6: a valid
/// citation "names an OPEN issue").
#[test]
fn prose_citation_without_issue_ref_fails() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void plain(chelis_string s);".to_string(),
        flags: vec![],
        citation: "reviewed and fine".to_string(),
    };
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(err.contains("CITATION NAMES NO ISSUE"), "{err}");
}

/// A maintainer override must name its issue too (§C6: "naming its
/// reason and issue"), or the liveness gate has nothing to hold it to.
#[test]
fn maintainer_override_without_issue_ref_fails() {
    let row = flagged_row(
        "planted.h: void staged(int out_dtype);",
        "maintainer-override(because I said so)",
    );
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(err.contains("CITATION NAMES NO ISSUE"), "{err}");
}

/// Unflagged rows keep the open-issue path (invariant 7's release
/// policy governs those additions, not the seam freeze).
#[test]
fn unflagged_row_with_issue_citation_passes() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void plain(chelis_string s);".to_string(),
        flags: vec![],
        citation: "chelis#123456".to_string(),
    };
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    assert!(check_against_baseline(&[row], &baseline).is_ok());
}

// ---------------------------------------------------------------------------
// The re-red-team's four executed mutations (PR #950, 2026-07-30), kept as
// standing negative tests.
// ---------------------------------------------------------------------------

/// A macro-hidden float carrier must be visible: the census consumes the
/// REAL preprocessor's output, so the spelling arrives expanded.
#[test]
fn reviewer_preprocessor_capacity_seam_is_visible() {
    let dir = std::env::temp_dir().join(format!("census-pp-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp include dir");
    fs::write(
        dir.join("planted.h"),
        "#define CHELIS_NUM double\n\
         CHELIS_NUM chelis_macro_result(int64_t x);\n",
    )
    .expect("write planted header");
    let per_file = preprocessed_headers(&dir, &["planted.h"]);
    let text = per_file.get("planted.h").expect("planted attributed");
    let rows = header_rows("planted.h", text, &collect_typedefs(text));
    let row = rows
        .iter()
        .find(|r| r.id.contains("chelis_macro_result"))
        .expect("export enumerated");
    assert!(
        row.flags.iter().any(|f| f == "float-carrier"),
        "the preprocessed declaration carries double: {row:?}"
    );
    fs::remove_dir_all(&dir).ok();
}

/// A typedef-hidden raw dtype int must be visible: classification resolves
/// typedef spellings before matching.
#[test]
fn reviewer_typedef_capacity_seam_is_visible() {
    let rows = header_rows_local(
        "planted.h",
        "typedef int chelis_dtype_id;\n\
         void chelis_typedef_dtype(chelis_value value, chelis_dtype_id dtype);\n",
    );
    let row = rows
        .iter()
        .find(|r| r.id.contains("chelis_typedef_dtype"))
        .expect("export enumerated");
    assert!(
        row.flags.iter().any(|f| f == "raw-dtype-int"),
        "the typedef resolves to raw int: {row:?}"
    );
}

/// Removing one grandfathered seam and relocating its citation onto a
/// brand-new seam must fail even though the count stays constant: the
/// identity set is frozen in this file, not the regeneratable baseline.
#[test]
fn reviewer_grandfathered_identity_relocation_must_fail() {
    let mut rows: Vec<Row> = GRANDFATHER_SEAM_IDS
        .iter()
        .skip(1)
        .map(|id| flagged_row(id, GRANDFATHER_SEAM_CITATION))
        .collect();
    rows.push(flagged_row(
        "planted.h: void brand_new_seam(int output_dtype);",
        GRANDFATHER_SEAM_CITATION,
    ));
    let regenerated = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: rows.clone(),
    };
    let err = check_against_baseline(&rows, &regenerated).unwrap_err();
    assert!(
        err.contains("identity relocation") || err.contains("outside the frozen"),
        "the count-only freeze must not accept identity relocation: {err}"
    );
}

/// A numeric runtime export (exact-width types, so not a capacity seam)
/// cannot enter with only a tracker citation: it needs its spec/05
/// semantic registration in the same change set.
#[test]
fn reviewer_runtime_numeric_op_requires_semantic_registration() {
    let mut rows = header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);");
    assert_eq!(rows.len(), 1);
    assert!(!is_seam(&rows[0].flags), "exact-width types are not seams");
    assert!(
        rows[0].flags.iter().any(|f| f == "numeric-op"),
        "numeric-op membership is structural: {:?}",
        rows[0]
    );
    rows[0].citation = "chelis#729".to_string();
    let row = rows.remove(0);
    let regenerated = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(std::slice::from_ref(&row), &regenerated).unwrap_err();
    assert!(
        err.contains("NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION"),
        "a tracker citation is not a semantic decision: {err}"
    );
    let mut registered = row;
    registered.citation = "chelis#123456".to_string();
    let ok_baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![registered.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-1]",
    };
    assert!(
        check_against_baseline_with(
            &[registered],
            &ok_baseline,
            &[registration],
            "Synthetic controlling fixture:\n> **[05-OP-1]** Integer abs.",
        )
        .is_ok()
    );
}

// ---------------------------------------------------------------------------
// PR #956 correction tests (2026-07-30): these lock the exact omissions found
// by the exact-head review and the failed Linux Integration run.
// ---------------------------------------------------------------------------

#[test]
fn c_identity_is_token_canonical_across_preprocessor_whitespace() {
    let apple = header_rows_local(
        "planted.h",
        "chelis_string chelis_string_from_bool(_Bool value);\n",
    );
    let linux = header_rows_local(
        "planted.h",
        "chelis_string chelis_string_from_bool( _Bool value);\n",
    );
    assert_eq!(
        apple, linux,
        "C token identity must not depend on a preprocessor's whitespace rendering"
    );
}

#[test]
fn matched_row_float_carrier_metadata_change_fails() {
    let baseline_row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: int64_t f(int64_t value);".to_string(),
        flags: vec!["numeric-op".to_string()],
        citation: GRANDFATHER_PLAIN_CITATION.to_string(),
    };
    let mut current_row = baseline_row.clone();
    current_row.flags = vec!["float-carrier".to_string(), "numeric-op".to_string()];
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![baseline_row],
    };
    let err = check_against_baseline(&[current_row], &baseline).unwrap_err();
    assert!(
        err.contains("ENFORCEMENT METADATA CHANGED")
            && err.contains("float-carrier")
            && err.contains("numeric-op"),
        "matched identities must freeze their derived flags exactly: {err}"
    );
}

#[test]
fn matched_row_typedef_int64_to_double_metadata_change_fails() {
    let before = header_rows_local(
        "planted.h",
        "typedef int64_t planted_num;\nplanted_num f(planted_num value);\n",
    );
    let after = header_rows_local(
        "planted.h",
        "typedef double planted_num;\nplanted_num f(planted_num value);\n",
    );
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    assert_eq!(before[0].id, after[0].id, "typedef spelling stays stable");
    let mut baseline_row = before[0].clone();
    baseline_row.citation = GRANDFATHER_PLAIN_CITATION.to_string();
    let mut current_row = after[0].clone();
    current_row.citation = GRANDFATHER_PLAIN_CITATION.to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![baseline_row],
    };
    let err = check_against_baseline(&[current_row], &baseline).unwrap_err();
    assert!(
        err.contains("ENFORCEMENT METADATA CHANGED")
            && err.contains("float-carrier")
            && err.contains("numeric-op"),
        "typedef target changes must not evade the flag freeze: {err}"
    );
}

#[test]
fn matched_row_raw_dtype_metadata_change_fails() {
    let baseline_row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void f ( int dtype ) ;".to_string(),
        flags: vec![],
        citation: GRANDFATHER_PLAIN_CITATION.to_string(),
    };
    let mut current_row = baseline_row.clone();
    current_row.flags = vec!["raw-dtype-int".to_string()];
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![baseline_row],
    };
    let err = check_against_baseline(&[current_row], &baseline).unwrap_err();
    assert!(
        err.contains("ENFORCEMENT METADATA CHANGED") && err.contains("raw-dtype-int"),
        "raw-dtype classification is enforcement metadata: {err}"
    );
}

#[test]
fn unrelated_observation_atom_is_not_a_numeric_registration() {
    let mut rows = header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);");
    let mut row = rows.remove(0);
    row.citation = "chelis#729; spec/05-risc-primitives.md [05-OBS-1]".to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OBS-1]",
    };
    let spec = fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap();
    let err = check_against_baseline_with(&[row], &baseline, &[registration], &spec).unwrap_err();
    assert!(
        err.contains("NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION")
            && err.contains("wrong atom grammar/group")
            && err.contains("05-OBS-1"),
        "an unrelated existing atom must not bless a callable: {err}"
    );
}

#[test]
fn nonexistent_operation_atom_is_not_a_numeric_registration() {
    let mut row =
        header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    row.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-999]",
    };
    let err =
        check_against_baseline_with(&[row], &baseline, &[registration], "no op atoms").unwrap_err();
    assert!(
        err.contains("does not exist as a normative") && err.contains("[05-OP-999]"),
        "a syntactically valid but absent atom must fail: {err}"
    );
}

#[test]
fn operation_atom_cross_reference_is_not_a_normative_definition() {
    let mut row =
        header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    row.citation = "chelis#729".to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-1]",
    };
    let err = check_against_baseline_with(
        &[row],
        &baseline,
        &[registration],
        "A cross-reference to [05-OP-1] is not its definition.",
    )
    .unwrap_err();
    assert!(
        err.contains("does not exist as a normative"),
        "a textual mention must not satisfy atom existence: {err}"
    );
}

#[test]
fn registered_numeric_callable_still_requires_issue_reference() {
    let mut row =
        header_rows_local("planted.h", "int64_t chelis_abs_i64(int64_t value);").remove(0);
    row.citation = "reviewed semantic registration".to_string();
    let baseline = Baseline {
        version: 2,
        legs: coverage_manifest(),
        rows: vec![row.clone()],
    };
    let registration = SemanticRegistration {
        callable: "[header-export] planted.h: int64_t chelis_abs_i64 ( int64_t value ) ;",
        atom: "[05-OP-1]",
    };
    let err = check_against_baseline_with(
        &[row],
        &baseline,
        &[registration],
        "> **[05-OP-1]** Synthetic exact callable semantics.",
    )
    .unwrap_err();
    assert!(
        err.contains("CITATION NAMES NO ISSUE")
            && !err.contains("NUMERIC OP WITHOUT EXACT SEMANTIC REGISTRATION"),
        "registration and live-issue citation are independent obligations: {err}"
    );
}

fn planted_numeric_adt(variant_name: &str) -> Expr {
    let span = chelis_deep::Span::new(0, 0);
    let tprim = Expr::node(
        DeepTag::TPrim,
        Default::default(),
        vec![Expr::Atom(Atom::Symbol("f64".to_string()), span)],
        span,
    );
    let variant = Expr::node(
        DeepTag::Variant,
        Default::default(),
        vec![
            Expr::Atom(Atom::Symbol(variant_name.to_string()), span),
            tprim,
        ],
        span,
    );
    Expr::node(
        DeepTag::Deftype,
        Default::default(),
        vec![
            Expr::Atom(Atom::Symbol("Json".to_string()), span),
            Expr::List(List { elements: vec![] }, span),
            variant,
        ],
        span,
    )
}

#[test]
fn std_adt_identity_changes_when_same_dtype_variant_changes() {
    let mut before = Vec::new();
    let mut after = Vec::new();
    scan_deftypes(&[planted_numeric_adt("JNum")], "planted", &mut before);
    scan_deftypes(&[planted_numeric_adt("JNumber")], "planted", &mut after);
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    assert_ne!(
        before[0].id, after[0].id,
        "variant and field shape, not only the dtype set, controls ADT identity"
    );
}

#[test]
fn exported_public_numeric_stdlib_def_is_enumerated() {
    let decls = chelis_surf::parser::parse_str(
        "module Planted\n\
         export (public_numeric)\n\
         def public_numeric(x: int64) -> int64 = x\n",
    )
    .expect("planted stdlib source parses");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let mut rows = Vec::new();
    scan_deftypes(&exprs, "planted", &mut rows);
    assert!(
        rows.iter()
            .any(|row| { row.kind == "std-def-numeric" && row.id.contains("public_numeric") }),
        "every exported numeric stdlib def must have a semantic-registration row: {rows:?}"
    );
}

#[test]
fn conditional_public_abi_is_mechanically_rejected() {
    let dir = std::env::temp_dir().join(format!("census-context-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp include dir");
    fs::write(
        dir.join("planted.h"),
        "#ifdef PLANTED_WIDE\n\
         double chelis_contextual(double x);\n\
         #else\n\
         int64_t chelis_contextual(int64_t x);\n\
         #endif\n",
    )
    .expect("write planted header");
    let result = std::panic::catch_unwind(|| preprocessed_headers(&dir, &["planted.h"]));
    fs::remove_dir_all(&dir).ok();
    let panic = result.expect_err("context-varying public ABI must be rejected");
    let message = if let Some(s) = panic.downcast_ref::<String>() {
        s.as_str()
    } else if let Some(s) = panic.downcast_ref::<&str>() {
        s
    } else {
        ""
    };
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI"),
        "the rejection must teach the totality rule: {message}"
    );
}

#[test]
fn shared_header_cannot_have_multiple_public_macro_contexts() {
    let dir = std::env::temp_dir().join(format!("census-root-context-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp include dir");
    fs::write(
        dir.join("shared.h"),
        "CHELIS_NUM chelis_context_result(CHELIS_NUM value);\n",
    )
    .unwrap();
    fs::write(
        dir.join("a.h"),
        "#define CHELIS_NUM double\n#include \"shared.h\"\n",
    )
    .unwrap();
    fs::write(
        dir.join("b.h"),
        "#define CHELIS_NUM float\n#include \"shared.h\"\n",
    )
    .unwrap();
    let result = std::panic::catch_unwind(|| preprocessed_headers(&dir, &["a.h", "b.h"]));
    fs::remove_dir_all(&dir).ok();
    let panic = result.expect_err("multiple public macro contexts must be rejected");
    let message = if let Some(s) = panic.downcast_ref::<String>() {
        s.as_str()
    } else if let Some(s) = panic.downcast_ref::<&str>() {
        s
    } else {
        ""
    };
    assert!(
        message.contains("CONTEXT-VARYING PUBLIC ABI"),
        "the rejection must name the context-invariance policy: {message}"
    );
}

#[test]
fn coverage_legs_cannot_claim_covered_without_live_oracles() {
    let row = Row {
        kind: "header-export".to_string(),
        id: "planted.h: void plain(chelis_string s);".to_string(),
        flags: vec![],
        citation: GRANDFATHER_PLAIN_CITATION.to_string(),
    };
    let mut legs = coverage_manifest();
    legs.deferred
        .retain(|leg| leg.leg != "wire-schema-numeric-fields");
    legs.covered.push(CoveredLeg {
        leg: "wire-schema-numeric-fields".to_string(),
        artifact: "invented".to_string(),
        enumerator: "invented".to_string(),
        command: "invented".to_string(),
        expected_success: "invented".to_string(),
        mutations: vec!["invented".to_string()],
    });
    let baseline = Baseline {
        version: 2,
        legs,
        rows: vec![row.clone()],
    };
    let err = check_against_baseline(&[row], &baseline).unwrap_err();
    assert!(
        err.contains("INVALID COVERAGE MANIFEST")
            && err.contains("wire-schema-numeric-fields")
            && err.contains("enumerator")
            && err.contains("mutation_oracle"),
        "a prose relabel must not turn a deferred leg into covered: {err}"
    );
}
