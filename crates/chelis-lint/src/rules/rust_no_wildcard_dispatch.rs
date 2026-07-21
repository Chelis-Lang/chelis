//! Rule `rust-no-wildcard-dispatch` (spec/01-nomenclature.md §12.3;
//! `spec/design/loud_unsupported.md` section C4.2) - a blocking Rust-source
//! rule against catch-all `_ =>` match arms that MANUFACTURE a concrete
//! value of a configured closed dtype/IR enum.
//!
//! This is ratchet #2 of the chelis#730 loud-unsupported plan. The chelis#703
//! substitution class recurs when a stage answers an unsupported input by
//! defaulting a closed-enum dispatch to a plausible value - HIP's
//! `_ => ElemKind::F32` (census row 5), the arithmetic host-type default
//! `_ => HostType::Int64` (census row 7). rustc cannot forbid a wildcard;
//! this rule can. Adding a variant to a closed dtype enum should force a
//! dispatch decision at every site, not fall silently to a manufactured
//! default.
//!
//! # What it flags
//!
//! A top-level `_` / `_ if <guard>` match arm whose BODY constructs a
//! CONCRETE variant of a configured enum: `Prim::<V>`, `ElemKind::<V>`,
//! `RiscOp::<V>`, or `HostType::<V>` where `<V>` is an uppercase variant
//! name. `HostType::Unknown` is the one exclusion: it is the blessed
//! polymorphic marker (`spec/design/loud_unsupported.md` section C3 - legal
//! for genuinely polymorphic signatures, and it fails LOUD downstream at
//! the numeric-baking point, never a silent narrow), not a manufactured
//! concrete value.
//!
//! # What it does NOT flag (by design)
//!
//! - Classification filters that produce a non-enum value: `_ => None`,
//!   `_ => continue`, `_ => return None`, `_ => false`. These correctly say
//!   "not this case"; they do not fabricate a closed-enum value. Requiring
//!   exhaustiveness over the 52-variant `RiscOp` in every such filter is
//!   neither the chelis#703 class nor tractable.
//! - Loud invariant guards: `_ => unreachable!(...)`, `_ => panic!(...)`.
//!   A panic reachable from source is chelis#692's concern (section C1.3),
//!   not a value substitution. (Any dtype name inside the panic message is
//!   a string literal, which the scanner masks, so it never trips this
//!   rule.)
//! - Field/pass-through: `_ => node.output_type.precision` - no
//!   `Enum::Variant` construction.
//! - Further dispatch: `_ => match x { Enum::V => .. }` - the enum names in
//!   the inner arm PATTERNS construct nothing (pattern-position refs are
//!   excluded).
//! - IR generation: a `RiscOp` variant other than `Const` built in a
//!   wildcard fallback (`_ => dag.add_node(RiscOp::Load { .. }, ..)`) is
//!   real IR, not a substitution. The only flagged `RiscOp` shape is the
//!   silent zero-seed `RiscOp::Const { value: 0.0 }` (census row 16).
//!
//! The narrow, precise signal keeps the rule green on the current tree with
//! a single-digit allowlist of pre-existing legitimate keeps, while a
//! planted `_ => ElemKind::F32` / `_ => Prim::F32` / `_ => HostType::Int64`
//! goes red - the rule's negative test / mutation oracle.
//!
//! # Scope
//!
//! Only files under the configured lowering/emission crates
//! ([`CONFIGURED_CRATE_SRC`]). The token tripwire
//! (`crates/chelis-cli/tests/loud_unsupported_tripwire.rs`, section C4.3)
//! is the complementary bridge for generated-string contexts and the
//! `unwrap_or(...)` numeric-default spellings this AST-shape rule does not
//! see.
//!
//! # Allowlist
//!
//! [`ALLOWLIST`] is a per-`(file, enum::variant)` count baseline, each with
//! a written justification, mirroring the tripwire's proven robust-to-line-
//! shift design. A file may carry up to its baseline count of a given
//! manufactured token; the FIRST unrecorded occurrence pushes the count
//! over and goes red. Adding a keep is a reviewable allowlist edit with a
//! justification (section C4.2), never a silent config change.

use std::collections::BTreeMap;

use crate::{Context, Rule, Severity, Surface, Violation};

/// Repo-relative (`/`-separated) `src/` prefixes this rule scans. The
/// lowering + all three backends + the compiler-api numeric modules - the
/// crates that dispatch on `Prim`/`RiscOp`/`ElemKind`/`HostType` (section
/// C4.2's crate list). `DeepTag` (chelis#731 Phase 3) will join
/// [`CONFIGURED_ENUMS`] without touching this list.
const CONFIGURED_CRATE_SRC: &[&str] = &[
    "crates/chelis-ir/src/",
    "crates/chelis-backend-c/src/",
    "crates/chelis-backend-hip/src/",
    "crates/chelis-backend-metal/src/",
    "crates/chelis-compiler-api/src/",
];

/// The closed enums whose manufactured-in-a-wildcard variants are the
/// chelis#703 substitution shape. `HostType::Unknown` is exempt in
/// [`is_concrete_construction`].
const CONFIGURED_ENUMS: &[&str] = &["Prim", "ElemKind", "RiscOp", "HostType"];

/// One allowlisted keep: `(repo-relative path, enum::variant token, count,
/// justification)`. The count is how many wildcard arms in that file may
/// manufacture that token; a NEW one trips the rule. Every entry is a
/// documented structural keep, not a chelis#703 substitution.
type Allow = (&'static str, &'static str, usize, &'static str);

const ALLOWLIST: &[Allow] = &[
    (
        "crates/chelis-ir/src/host.rs",
        "HostType::Bool",
        1,
        "builtin_result_host_type inference: a non-tensor `cmplt` yields \
         bool by construction (its result type), not an unsupported-case \
         default",
    ),
    (
        "crates/chelis-ir/src/host.rs",
        "HostType::Float64",
        2,
        "builtin_result_host_type inference: `tensor_to_scalar` and the \
         float-precision class classify coarsely to Float64 (Issue #308); \
         non-float precisions are rejected upstream, never silently \
         defaulted here",
    ),
    (
        "crates/chelis-ir/src/host.rs",
        "HostType::List",
        1,
        "an empty `Nil` list literal whose declared type is not a List gets \
         the coarse `List<Unknown>` element type - the static type of an \
         empty list, not a value substitution",
    ),
    (
        "crates/chelis-backend-c/src/host_emit.rs",
        "HostType::List",
        1,
        "ownership-tracking type for track_owned_alloc when the static type \
         is not a List (scope-exit release bookkeeping, issue #406); \
         metadata for the release pass, not an emitted value",
    ),
    (
        "crates/chelis-backend-c/src/host_emit.rs",
        "HostType::Tuple",
        1,
        "ownership-tracking type for track_owned_alloc when the static type \
         is not a Tuple (issue #406/#310); metadata for the release pass, \
         not an emitted value",
    ),
    (
        "crates/chelis-ir/src/lower.rs",
        "RiscOp::Const",
        1,
        "census row 16 keep (section C1.4): the unknown-tag fallthrough \
         sequence-lowers children and the CHILDLESS case raises \
         (raise_malformed_deep); the Const seed is a sequencing no-op kept \
         dead by the parser's closed 62-tag vocabulary, never a value \
         substitution",
    ),
];

pub struct RustNoWildcardDispatch;

impl Rule for RustNoWildcardDispatch {
    fn id(&self) -> &str {
        "rust-no-wildcard-dispatch"
    }

    fn spec_ref(&self) -> &str {
        "§12.3"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::RustSource]
    }

    fn summary(&self) -> &str {
        "no catch-all `_ =>` match arm may manufacture a concrete value of a \
         closed dtype/IR enum (Prim/ElemKind/RiscOp/HostType) in the \
         lowering/emission crates; convert to an exhaustive match or a \
         branded unsupported rejection"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let Some(rel) = repo_relative(ctx) else {
            return Vec::new();
        };
        if !CONFIGURED_CRATE_SRC.iter().any(|p| rel.starts_with(*p)) {
            return Vec::new();
        }

        let masked = mask_strings_and_comments(source);
        let arms = wildcard_substitution_arms(&masked, source);

        // Group by the manufactured `enum::variant` token, then diff each
        // group against its per-file allowlist baseline. A file may carry
        // up to its baseline count; anything above it is a violation.
        let mut by_token: BTreeMap<&str, Vec<Hit>> = BTreeMap::new();
        for hit in arms {
            by_token.entry(hit.token).or_default().push(hit);
        }

        let mut out = Vec::new();
        for (token, hits) in by_token {
            let allowed = ALLOWLIST
                .iter()
                .find(|(path, tok, _, _)| *path == rel.as_str() && *tok == token)
                .map(|(_, _, count, _)| *count)
                .unwrap_or(0);
            if hits.len() <= allowed {
                continue;
            }
            for hit in hits {
                out.push(Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(hit.line),
                    col: Some(hit.col),
                    message: format!(
                        "catch-all `_` arm manufactures `{token}` - a concrete \
                         closed-enum value defaulted on the unsupported branch \
                         (chelis#703 class). Make the match exhaustive over the \
                         enum, or reject with a branded `unsupported:` \
                         diagnostic (section C2). A documented structural keep \
                         goes in ALLOWLIST in rust_no_wildcard_dispatch.rs with \
                         a justification (this file allows {allowed})."
                    ),
                });
            }
        }
        out.sort_by_key(|v| v.line);
        out
    }
}

struct Hit {
    line: usize,
    col: usize,
    token: &'static str,
}

fn repo_relative(ctx: &Context<'_>) -> Option<String> {
    let rel = ctx.path.strip_prefix(ctx.root).ok()?;
    Some(
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

/// Return a length-preserving copy of `src` with the CONTENTS of string
/// literals, char literals, and comments replaced by spaces. Structural
/// characters (braces, parens, `=>`, `::`) are preserved, so downstream
/// byte offsets map back to the original source unchanged.
fn mask_strings_and_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out: Vec<u8> = src.bytes().collect();
    let mut i = 0usize;
    let blank = |out: &mut Vec<u8>, at: usize| {
        if out[at] != b'\n' {
            out[at] = b' ';
        }
    };
    while i < n {
        let c = bytes[i];
        // Line comment.
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            while i < n && bytes[i] != b'\n' {
                blank(&mut out, i);
                i += 1;
            }
            continue;
        }
        // Block comment (non-nesting is fine for masking).
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            blank(&mut out, i);
            blank(&mut out, i + 1);
            i += 2;
            while i < n && !(bytes[i] == b'*' && i + 1 < n && bytes[i + 1] == b'/') {
                blank(&mut out, i);
                i += 1;
            }
            if i < n {
                blank(&mut out, i);
                if i + 1 < n {
                    blank(&mut out, i + 1);
                }
                i += 2;
            }
            continue;
        }
        // Raw string: r"..." or r#"..."# ... "#.
        if c == b'r' && i + 1 < n && (bytes[i + 1] == b'"' || bytes[i + 1] == b'#') {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < n && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < n && bytes[j] == b'"' {
                j += 1;
                // Scan to the closing `"` followed by `hashes` `#`.
                loop {
                    if j >= n {
                        break;
                    }
                    if bytes[j] == b'"' {
                        let mut k = j + 1;
                        let mut got = 0usize;
                        while k < n && got < hashes && bytes[k] == b'#' {
                            got += 1;
                            k += 1;
                        }
                        if got == hashes {
                            // Blank the whole raw string incl. delimiters.
                            for m in i..k.min(n) {
                                blank(&mut out, m);
                            }
                            j = k;
                            break;
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        // Normal string.
        if c == b'"' {
            blank(&mut out, i);
            i += 1;
            while i < n && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    blank(&mut out, i);
                    i += 1;
                    if i < n {
                        blank(&mut out, i);
                        i += 1;
                    }
                    continue;
                }
                blank(&mut out, i);
                i += 1;
            }
            if i < n {
                blank(&mut out, i);
                i += 1;
            }
            continue;
        }
        // Char literal 'x' / '\n' vs lifetime 'a. A char literal has a
        // closing quote within 3-4 bytes; a lifetime does not.
        if c == b'\'' {
            if i + 2 < n && bytes[i + 1] == b'\\' {
                // '\?...' escape - blank until closing quote.
                let mut j = i + 1;
                while j < n && bytes[j] != b'\'' {
                    j += 1;
                }
                for m in i..=(j.min(n - 1)) {
                    blank(&mut out, m);
                }
                i = j + 1;
                continue;
            }
            if i + 2 < n && bytes[i + 2] == b'\'' {
                // 'x' single char.
                blank(&mut out, i);
                blank(&mut out, i + 1);
                blank(&mut out, i + 2);
                i += 3;
                continue;
            }
            // Lifetime or label; leave as-is.
            i += 1;
            continue;
        }
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

/// Byte ranges `(open_brace, close_brace)` of each `match` arm block.
fn match_arm_blocks(code: &str) -> Vec<(usize, usize)> {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < n {
        if is_keyword_at(bytes, i, b"match") {
            // Scan for the arm-block `{` at bracket depth 0 (parens/brackets
            // in the scrutinee are balanced; a struct-literal scrutinee must
            // be parenthesized, so the first depth-0 `{` is the arm block).
            let mut j = i + 5;
            let mut depth = 0i32;
            let mut open = None;
            while j < n {
                match bytes[j] {
                    b'(' | b'[' => depth += 1,
                    b')' | b']' => depth -= 1,
                    b'{' if depth == 0 => {
                        open = Some(j);
                        break;
                    }
                    b';' if depth == 0 => break,
                    _ => {}
                }
                j += 1;
            }
            if let Some(open) = open {
                // Matching close brace.
                let mut bdepth = 0i32;
                let mut k = open;
                while k < n {
                    match bytes[k] {
                        b'{' => bdepth += 1,
                        b'}' => {
                            bdepth -= 1;
                            if bdepth == 0 {
                                blocks.push((open, k));
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                i = open + 1;
                continue;
            }
        }
        i += 1;
    }
    blocks
}

fn is_keyword_at(bytes: &[u8], i: usize, kw: &[u8]) -> bool {
    if !bytes[i..].starts_with(kw) {
        return false;
    }
    let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
    let after = i + kw.len();
    let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
    before_ok && after_ok
}

fn is_ident_byte(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

/// Parse the top-level arms of a match block and return each wildcard arm
/// that manufactures a concrete configured-enum variant.
fn wildcard_substitution_arms(masked: &str, original: &str) -> Vec<Hit> {
    let mut hits = Vec::new();
    for (open, close) in match_arm_blocks(masked) {
        for arm in top_level_arms(masked, open, close) {
            if !is_wildcard_pattern(&masked[arm.pat_start..arm.arrow]) {
                continue;
            }
            let body = &masked[arm.arrow + 2..arm.body_end];
            if let Some(token) = concrete_construction(body) {
                let (line, col) = line_col(original, arm.arrow);
                hits.push(Hit { line, col, token });
            }
        }
    }
    hits
}

struct ArmSpan {
    pat_start: usize,
    arrow: usize,
    body_end: usize,
}

/// Split a match block's interior into top-level arms. Handles both
/// comma-terminated expression arms (`P => expr,`) and brace-block arms
/// (`P => { .. }` with an optional trailing comma).
fn top_level_arms(code: &str, open: usize, close: usize) -> Vec<ArmSpan> {
    let bytes = code.as_bytes();
    let mut arms = Vec::new();
    let mut i = open + 1;
    while i < close {
        // Skip leading whitespace and stray commas.
        while i < close && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= close {
            break;
        }
        let pat_start = i;
        // Scan to the top-level `=>`.
        let mut depth = 0i32;
        let mut arrow = None;
        while i < close {
            match bytes[i] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b'=' if depth == 0 && i + 1 < close && bytes[i + 1] == b'>' => {
                    arrow = Some(i);
                    break;
                }
                _ => {}
            }
            i += 1;
        }
        let Some(arrow) = arrow else { break };
        // Body starts after `=>`.
        i = arrow + 2;
        while i < close && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let body_end = if i < close && bytes[i] == b'{' {
            // Brace-block arm: end at the matching `}`.
            let mut bdepth = 0i32;
            let mut k = i;
            while k < close {
                match bytes[k] {
                    b'{' => bdepth += 1,
                    b'}' => {
                        bdepth -= 1;
                        if bdepth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            (k + 1).min(close)
        } else {
            // Expression arm: end at the next top-level comma.
            let mut d = 0i32;
            let mut k = i;
            while k < close {
                match bytes[k] {
                    b'(' | b'[' | b'{' => d += 1,
                    b')' | b']' | b'}' => d -= 1,
                    b',' if d == 0 => break,
                    _ => {}
                }
                k += 1;
            }
            k.min(close)
        };
        arms.push(ArmSpan {
            pat_start,
            arrow,
            body_end,
        });
        i = body_end;
    }
    arms
}

/// True if the arm pattern (text between the previous separator and `=>`)
/// is a bare wildcard, optionally with a match guard: `_` or `_ if <g>`.
fn is_wildcard_pattern(pat: &str) -> bool {
    let mut p = pat.trim();
    if let Some(rest) = p.strip_prefix('|') {
        p = rest.trim();
    }
    if let Some((head, _guard)) = p.split_once(" if ") {
        p = head.trim();
    }
    p == "_"
}

/// If `body` constructs a concrete configured-enum variant in EXPRESSION
/// position, return the `Enum::Variant` token. Returns `None` when the only
/// occurrences are exclusions:
///
/// - `HostType::Unknown` - the blessed polymorphic marker (§C3).
/// - a `RiscOp` variant other than `Const` - a `RiscOp::Load`/`MatMul`/...
///   built in a wildcard fallback is legitimate IR generation, not a
///   value substitution; the ONLY `RiscOp` substitution shape is the
///   silent zero-seed `RiscOp::Const { value: 0.0 }` (census row 16).
/// - a PATTERN-position ref (`Enum::Variant(..)? =>` / `... if`) - an inner
///   `match`/`if let` arm inside the wildcard body dispatches further; its
///   arm patterns name the enum but construct nothing.
fn concrete_construction(body: &str) -> Option<&'static str> {
    for &enm in CONFIGURED_ENUMS {
        let needle = format!("{enm}::");
        let mut from = 0usize;
        while let Some(rel) = body[from..].find(&needle) {
            let at = from + rel + needle.len();
            let variant: String = body[at..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            from = at;
            let Some(first) = variant.chars().next() else {
                continue;
            };
            if !first.is_ascii_uppercase() {
                continue; // an assoc fn / const (`Prim::parse_name`), not a variant.
            }
            if enm == "HostType" && variant == "Unknown" {
                continue;
            }
            if enm == "RiscOp" && variant != "Const" {
                continue;
            }
            if in_pattern_position(body, at + variant.len()) {
                continue;
            }
            return Some(intern_token(enm, &variant));
        }
    }
    None
}

/// True if the enum ref whose variant ends at byte `pos` is a match/`if let`
/// PATTERN rather than a constructed value: after the optional
/// variant-args group it is immediately followed by `=>` or a match guard
/// `if `.
fn in_pattern_position(body: &str, pos: usize) -> bool {
    let bytes = body.as_bytes();
    let n = body.len();
    let mut i = pos;
    while i < n && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    // Skip a balanced tuple/struct args group `(..)` or `{..}`.
    if i < n && (bytes[i] == b'(' || bytes[i] == b'{') {
        let (open, close) = if bytes[i] == b'(' {
            (b'(', b')')
        } else {
            (b'{', b'}')
        };
        let mut depth = 0i32;
        while i < n {
            if bytes[i] == open {
                depth += 1;
            } else if bytes[i] == close {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            i += 1;
        }
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
    }
    body[i..].starts_with("=>") || body[i..].starts_with("if ")
}

/// Map an `(enum, variant)` pair to a `'static` token for grouping and
/// allowlist keying. Only the tokens the tree/allowlist reference need to
/// resolve to a stable string; anything else (a genuinely new manufactured
/// variant) interns to the enum name, which still has an allowlist baseline
/// of 0 and therefore trips the rule.
fn intern_token(enm: &str, variant: &str) -> &'static str {
    match (enm, variant) {
        ("Prim", "F32") => "Prim::F32",
        ("Prim", "F64") => "Prim::F64",
        ("Prim", "Int64") => "Prim::Int64",
        ("Prim", "Int32") => "Prim::Int32",
        ("Prim", "Bool") => "Prim::Bool",
        ("ElemKind", "F32") => "ElemKind::F32",
        ("ElemKind", "F64") => "ElemKind::F64",
        ("RiscOp", "Const") => "RiscOp::Const",
        ("HostType", "Bool") => "HostType::Bool",
        ("HostType", "Float64") => "HostType::Float64",
        ("HostType", "Int64") => "HostType::Int64",
        ("HostType", "List") => "HostType::List",
        ("HostType", "Tuple") => "HostType::Tuple",
        ("Prim", _) => "Prim::<variant>",
        ("ElemKind", _) => "ElemKind::<variant>",
        ("RiscOp", _) => "RiscOp::<variant>",
        ("HostType", _) => "HostType::<variant>",
        _ => "<enum>::<variant>",
    }
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if index >= offset {
            break;
        }
        if byte == b'\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    (line, offset.saturating_sub(line_start) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn ctx<'a>(rel: &'a str, src: &'a str) -> Context<'a> {
        // `root` is `/repo`; `path` is `/repo/<rel>` so `repo_relative`
        // yields `rel`.
        Context {
            root: Path::new("/repo"),
            path: Path::new(leak(format!("/repo/{rel}"))),
            source: Some(src),
            surface: Surface::RustSource,
        }
    }

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    // ---- the mutation oracle: a planted wildcard goes red -----------------

    #[test]
    fn flags_planted_elemkind_f32_default() {
        let src = "fn k(p: Prim) -> ElemKind {\n    match p {\n        \
                   Prim::F32 => ElemKind::F32,\n        _ => ElemKind::F32,\n    }\n}\n";
        let v = RustNoWildcardDispatch
            .check(&ctx("crates/chelis-backend-hip/src/emit.rs", src));
        assert_eq!(v.len(), 1, "planted `_ => ElemKind::F32` must fire");
        assert_eq!(v[0].rule_id, "rust-no-wildcard-dispatch");
        assert_eq!(v[0].spec_ref, "§12.3");
    }

    #[test]
    fn flags_planted_prim_f32_default() {
        let src = "fn f(x: HostType) -> Prim {\n    match x {\n        \
                   HostType::Int64 => Prim::Int64,\n        _ => Prim::F32,\n    }\n}\n";
        let v = RustNoWildcardDispatch
            .check(&ctx("crates/chelis-ir/src/specialize.rs", src));
        assert_eq!(v.len(), 1, "planted `_ => Prim::F32` must fire");
    }

    #[test]
    fn flags_planted_hosttype_int64_arith_default() {
        // Census row 7's shape.
        let src = "fn t(a: HostType) -> HostType {\n    match a {\n        \
                   HostType::Float64 => HostType::Float64,\n        \
                   _ => HostType::Int64,\n    }\n}\n";
        let v = RustNoWildcardDispatch
            .check(&ctx("crates/chelis-ir/src/host.rs", src));
        assert!(
            v.iter().any(|x| x.message.contains("HostType::Int64")),
            "planted `_ => HostType::Int64` must fire; got {v:?}"
        );
    }

    #[test]
    fn flags_wildcard_with_guard() {
        let src = "fn f(p: Prim) -> ElemKind {\n    match p {\n        \
                   Prim::F64 => ElemKind::F64,\n        _ if true => ElemKind::F32,\n    }\n}\n";
        let v = RustNoWildcardDispatch
            .check(&ctx("crates/chelis-backend-metal/src/emit.rs", src));
        assert_eq!(v.len(), 1, "`_ if <guard>` wildcard must fire too");
    }

    #[test]
    fn flags_block_bodied_wildcard() {
        let src = "fn f(p: Prim) -> ElemKind {\n    match p {\n        \
                   Prim::F64 => ElemKind::F64,\n        _ => {\n            \
                   let k = ElemKind::F32;\n            k\n        }\n    }\n}\n";
        let v = RustNoWildcardDispatch
            .check(&ctx("crates/chelis-backend-hip/src/emit.rs", src));
        assert_eq!(v.len(), 1, "a block-bodied wildcard manufacturing must fire");
    }

    // ---- negative parity: the safe shapes stay green ----------------------

    #[test]
    fn ignores_classification_none_filter() {
        // The dominant RiscOp shape: `_ => None` is not a substitution.
        let src = "fn is_reduce(op: &RiscOp) -> Option<u8> {\n    match op {\n        \
                   RiscOp::Sum => Some(0),\n        _ => None,\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/lower.rs", src))
                .is_empty(),
            "`_ => None` classification must not fire"
        );
    }

    #[test]
    fn ignores_unreachable_and_continue_guards() {
        let src = "fn f(op: &RiscOp) {\n    match op {\n        \
                   RiscOp::Sum => {}\n        _ => unreachable!(\"ElemKind::F32 in a string\"),\n    }\n    \
                   match op {\n        RiscOp::Max => {}\n        _ => return,\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-backend-hip/src/emit.rs", src))
                .is_empty(),
            "unreachable!/return wildcards (even with a dtype name inside a \
             string) must not fire"
        );
    }

    #[test]
    fn ignores_hosttype_unknown_polymorphic_marker() {
        // The blessed polymorphic default (section C3) - stays green.
        let src = "fn f(a: Option<HostType>) -> Option<HostType> {\n    match a {\n        \
                   Some(HostType::Tensor(t)) => Some(HostType::Tensor(t)),\n        \
                   _ => Some(HostType::Unknown),\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/host.rs", src))
                .is_empty(),
            "`_ => Some(HostType::Unknown)` must not fire (blessed marker)"
        );
    }

    #[test]
    fn ignores_field_passthrough() {
        let src = "fn f(op: &RiscOp, n: &Node) -> Prim {\n    match op {\n        \
                   RiscOp::Sum => Prim::F32,\n        _ => n.output_type.precision,\n    }\n}\n";
        // The `RiscOp::Sum => Prim::F32` is a NAMED arm, not a wildcard; the
        // `_` arm is a field pass-through. No wildcard manufactures a value.
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/specialize.rs", src))
                .is_empty(),
            "a field-access pass-through wildcard must not fire"
        );
    }

    #[test]
    fn ignores_inner_match_pattern_refs() {
        // A wildcard whose body dispatches FURTHER through a nested match
        // names the enum in the inner arm PATTERNS - it constructs nothing.
        // (The host.rs `lookup_access_field` shape.)
        let src = "fn f(base: &X) -> Option<u8> {\n    match base {\n        \
                   X::A => Some(0),\n        _ => match host_type(base) {\n            \
                   HostType::Adt(name, args) => lookup(name, args),\n            \
                   _ => None,\n        },\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/host.rs", src))
                .is_empty(),
            "an enum ref in a nested-match PATTERN must not be read as a \
             construction"
        );
    }

    #[test]
    fn riscop_ir_gen_is_ignored_but_const_zero_seed_flags() {
        // A wildcard fallback building a real IR node (Load/MatMul) is IR
        // generation, not a value substitution - not flagged.
        let load = "fn l(name: &str) -> Node {\n    match name {\n        \
                    \"mm\" => make(RiscOp::MatMul),\n        \
                    _ => dag.add_node(RiscOp::Load { name: name.into() }, vec![]),\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/lower.rs", load))
                .is_empty(),
            "a `_ => RiscOp::Load(...)` IR-gen fallback must not fire"
        );
        // The silent zero-seed shape (census row 16) IS the RiscOp
        // substitution and DOES fire (planted a second time, since lower.rs
        // allowlists exactly one).
        let seed = "fn s(tag: &str) -> V {\n    match tag {\n        \
                    \"a\" => real(),\n        \
                    _ => Node(dag.add_node(RiscOp::Const { value: 0.0 }, vec![])),\n    }\n}\n\
                    fn s2(tag: &str) -> V {\n    match tag {\n        \
                    \"a\" => real(),\n        \
                    _ => Node(dag.add_node(RiscOp::Const { value: 0.0 }, vec![])),\n    }\n}\n";
        assert!(
            !RustNoWildcardDispatch
                .check(&ctx("crates/chelis-ir/src/lower.rs", seed))
                .is_empty(),
            "two `_ => RiscOp::Const {{ 0.0 }}` seeds exceed the single \
             allowlisted keep and must fire"
        );
    }

    #[test]
    fn does_not_scan_unconfigured_crates() {
        let src = "fn k(p: Prim) -> ElemKind {\n    match p {\n        _ => ElemKind::F32,\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-lint/src/lib.rs", src))
                .is_empty(),
            "files outside the configured crate list are not scanned"
        );
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-cli/tests/loud_unsupported_tripwire.rs", src))
                .is_empty(),
            "the tripwire test's own planted strings are out of scope"
        );
    }

    #[test]
    fn allowlisted_keep_stays_green_but_a_second_goes_red() {
        // One `HostType::List` wildcard is allowlisted in host_emit.rs; a
        // planted SECOND one pushes the count over the baseline.
        let one = "fn a() -> HostType {\n    match t {\n        \
                   HostType::List(x) => HostType::List(x),\n        \
                   _ => HostType::List(Box::new(HostType::Unknown)),\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-backend-c/src/host_emit.rs", one))
                .is_empty(),
            "the single allowlisted HostType::List keep stays green"
        );
        let two = format!("{one}{one}");
        assert!(
            !RustNoWildcardDispatch
                .check(&ctx("crates/chelis-backend-c/src/host_emit.rs", &two))
                .is_empty(),
            "a second unrecorded HostType::List wildcard trips the baseline"
        );
    }

    #[test]
    fn string_masking_prevents_false_positive() {
        // A dtype construction inside a string literal is not real code.
        let src = "fn f(op: &RiscOp) -> String {\n    match op {\n        \
                   RiscOp::Sum => String::new(),\n        \
                   _ => \"HostType::Int64\".to_string(),\n    }\n}\n";
        assert!(
            RustNoWildcardDispatch
                .check(&ctx("crates/chelis-backend-c/src/host_emit.rs", src))
                .is_empty(),
            "an enum name inside a string literal must not fire"
        );
    }
}
