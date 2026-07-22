//! chelis#730 Phase 0 - the token tripwire (spec/design/loud_unsupported.md
//! section C4.3).
//!
//! A fast source-tree grep over the recidivist tokens of the chelis#703
//! class (unsupported cases silently substitute a value). The verified
//! section C5 census is frozen here as `BASELINE`; any NEW occurrence of a
//! recidivist token goes red with a pointer back at the plan. This exists
//! as supporting source-inventory evidence while the section C3 Result channel
//! and the section C4 typed boundaries land. It is never the structural
//! authority.
//!
//! Also hosted here on behalf of the chelis#732 plan's Phase 0 (its item 3:
//! "coordinate, do not duplicate", per the roadmap's Wave 0 handshake): the
//! `%.16g` / `%.1f` / `{value:.1}` format-token row, the observation-channel
//! narrowing exits. That row's new-site violations point at
//! `spec/design/faithful_observation.md` B2.4 and its production allowlist
//! only ever shrinks (deleted entirely by #732 Phase 2).
//!
//! Baseline maintenance (B1/B2.5 of the plan):
//! - a count INCREASE is either a regression (remove the token) or new
//!   work: file an issue, append a section C5 census row, and extend
//!   `BASELINE` in the same change set - never silently;
//! - a count DECREASE accompanies the site's fix: shrink `BASELINE` in the
//!   fixing PR (removals only with the site's fix). For BASELINE entries
//!   with no census row (the annotated benign carriers, e.g. host.rs's
//!   `unwrap_or_default()`s), a decrease just means shrink the baseline
//!   in the same PR - there is no censused "fix" to ride with.
//!
//! Known limits of this source inventory (PR #746 review):
//! - FALSE-RED surface: `CFormatNarrowing` scans ALL crates/*/src for the
//!   very common `%.1f`/`%.16g` tokens. A NEW legitimate use anywhere
//!   (a timing print, a benchmark) trips the increase branch; if it is
//!   genuinely not a numeric-observation exit, extend `BASELINE` with an
//!   annotation in the same PR - the faithful_observation.md B2.4
//!   pointer in the message applies only to formatter sites.
//! - FALSE-GREEN evasions (grep is textual): `*/ 0.0"` / `*/0"` /
//!   unquoted stubs evade `StubZero`; a multi-line `_ =>` arm evades
//!   `ElemKindWildcardArm` (same-line match only), as do NAMED catch-alls
//!   (`other =>`) - though a named catch-all emitting a stub is caught by
//!   `StubZero`, per census row 19; non-quote-adjacent `<value>`
//!   spellings evade `ValuePlaceholder`. The typed boundaries and their
//!   mutation oracles close the semantic class; do not treat this inventory as
//!   airtight.

#![allow(clippy::uninlined_format_args)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The recidivist token classes from section C4.3 of the plan, plus the
/// hosted chelis#732 format row.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Pat {
    /// `*/ 0"` - a `/* unsupported ... */ 0` stub inside an emitted-string
    /// builder (census rows 2 and 19).
    StubZero,
    /// `"<value>` - the quoted placeholder literal the C emitter prints for
    /// an unclassifiable or unsupported value (census rows 3 and 4).
    ValuePlaceholder,
    /// `unwrap_or_default()` in lowering/emission crates (census row 8's
    /// mechanism: defaulted extraction results become silent no-ops).
    UnwrapOrDefault,
    /// `unwrap_or(Prim::` in lowering/emission/runtime-packing paths
    /// (census rows 13 and 14: a missing precision becomes F32).
    UnwrapOrPrim,
    /// A wildcard match arm producing an `ElemKind` (census row 5: HIP's
    /// `_ => kernels::ElemKind::F32` dtype substitution).
    ElemKindWildcardArm,
    /// A numeric-literal default on a NON-COMMENT line in
    /// lowering/emission/runtime-packing paths - the value-default shape
    /// chelis#776 exposed (a user value silently replaced by a baked
    /// default). Added at Phase 1 per the census-maintenance obligation;
    /// the baseline annotates each surviving site as proven-structural
    /// or P1-frozen for the Phase 2 typed-boundary audit. Phase 2 (rt791 F6) widens
    /// the class beyond `.unwrap_or(<lit>)` to the closure/`map_or`
    /// spellings that evaded it: `.unwrap_or_else(|| <lit>)` and
    /// `.map_or(<lit>, ...)` - the same silent numeric default wearing a
    /// closure. This token class remains supporting evidence for numeric
    /// default spellings; the typed boundary is the authority.
    UnwrapOrNumericLiteral,
    /// `%.16g` / `%.1f` / `{value:.1}` anywhere in crate sources - the
    /// f64-shaped observation-channel exits (chelis#716/#723/#728; owned by
    /// the chelis#732 plan, hosted in this tripwire per its Phase 0 item 3
    /// and the Wave 0 handshake; a new site violates
    /// spec/design/faithful_observation.md B2.4, no third formatter).
    CFormatNarrowing,
}

const ALL_PATS: &[Pat] = &[
    Pat::StubZero,
    Pat::ValuePlaceholder,
    Pat::UnwrapOrDefault,
    Pat::UnwrapOrPrim,
    Pat::ElemKindWildcardArm,
    Pat::UnwrapOrNumericLiteral,
    Pat::CFormatNarrowing,
];

impl Pat {
    fn id(self) -> &'static str {
        match self {
            Pat::StubZero => "silent-stub-zero",
            Pat::ValuePlaceholder => "value-placeholder",
            Pat::UnwrapOrDefault => "unwrap-or-default",
            Pat::UnwrapOrPrim => "unwrap-or-prim",
            Pat::ElemKindWildcardArm => "elemkind-wildcard-arm",
            Pat::UnwrapOrNumericLiteral => "unwrap-or-numeric-literal",
            Pat::CFormatNarrowing => "c-format-narrowing",
        }
    }

    /// Path prefixes (relative to the repo root, `/`-separated) this
    /// pattern is scanned under. Section C4.3 scopes the unwrap tokens to
    /// lowering/emission paths; the string tokens are global.
    fn scopes(self) -> &'static [&'static str] {
        const ALL_CRATE_SRC: &[&str] = &["crates/"];
        const LOWERING_AND_EMISSION: &[&str] = &[
            "crates/chelis-ir/src/",
            "crates/chelis-backend-c/src/",
            "crates/chelis-backend-hip/src/",
            "crates/chelis-backend-metal/src/",
        ];
        const LOWERING_EMISSION_AND_RUNTIME_PACKING: &[&str] = &[
            "crates/chelis-ir/src/",
            "crates/chelis-backend-c/src/",
            "crates/chelis-backend-hip/src/",
            "crates/chelis-backend-metal/src/",
            "crates/chelis-compiler-api/src/",
        ];
        const BACKENDS: &[&str] = &[
            "crates/chelis-backend-c/src/",
            "crates/chelis-backend-hip/src/",
            "crates/chelis-backend-metal/src/",
        ];
        match self {
            Pat::StubZero | Pat::ValuePlaceholder | Pat::CFormatNarrowing => ALL_CRATE_SRC,
            Pat::UnwrapOrDefault => LOWERING_AND_EMISSION,
            Pat::UnwrapOrPrim | Pat::UnwrapOrNumericLiteral => {
                LOWERING_EMISSION_AND_RUNTIME_PACKING
            }
            Pat::ElemKindWildcardArm => BACKENDS,
        }
    }

    /// The design doc a NEW occurrence of this pattern violates.
    fn doc(self) -> &'static str {
        match self {
            Pat::CFormatNarrowing => {
                "spec/design/faithful_observation.md B2.4 (no third formatter)"
            }
            _ => "spec/design/loud_unsupported.md B2.5",
        }
    }

    fn count(self, content: &str) -> usize {
        fn occurrences(content: &str, token: &str) -> usize {
            content.matches(token).count()
        }
        match self {
            Pat::StubZero => occurrences(content, "*/ 0\""),
            Pat::ValuePlaceholder => occurrences(content, "\"<value>"),
            Pat::UnwrapOrDefault => occurrences(content, "unwrap_or_default()"),
            Pat::UnwrapOrPrim => occurrences(content, "unwrap_or(Prim::"),
            Pat::ElemKindWildcardArm => content
                .lines()
                .filter(|line| {
                    let t = line.trim_start();
                    t.starts_with("_ =>") && t.contains("ElemKind::")
                })
                .count(),
            // Comment lines are skipped so a doc reference to the banned
            // token (e.g. a conversion comment quoting the pre-fix code)
            // does not count as a live site; the section C4.2 lint closes
            // string/comment evasions structurally at Phase 2. The three
            // spellings counted are the direct default `.unwrap_or(<lit>)`,
            // the closure default `.unwrap_or_else(|| <lit>)`, and the
            // map-or default `.map_or(<lit>, ...)` - rt791 F6 (Phase 2)
            // added the latter two, which wore a closure to evade the
            // original single-spelling scan.
            Pat::UnwrapOrNumericLiteral => content
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .map(count_numeric_defaults)
                .sum(),
            Pat::CFormatNarrowing => {
                occurrences(content, "%.16g")
                    + occurrences(content, "%.1f")
                    + occurrences(content, "{value:.1}")
            }
        }
    }
}

/// One frozen baseline entry: (pattern, repo-relative path, occurrence
/// count, why it is allowed to exist). The census row or issue in the
/// annotation is the audit trail; entries without a census row are frozen
/// as-is at the Phase 0 baseline and fall to the Phase 2 lint audit.
type Entry = (Pat, &'static str, usize, &'static str);

const BASELINE: &[Entry] = &[
    // -- silent-stub-zero: ZERO entries left - census rows 2 and 19
    // converted at Phase 1 (the stub arm is Err(Unsupported); Metal's
    // host_scalar_literal rejects through its String channel) -----------
    // -- value-placeholder: ZERO entries left - census rows 3 and 4
    // converted at Phase 1 (both print arms and the to_string catch-all
    // are Err(Unsupported)) ---------------------------------------------
    // -- unwrap-or-default --------------------------------------------------
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-ir/src/lower.rs",
        5,
        "census row 8's two reduce_window extraction uses converted at \
         Phase 1 (chelis#725); the remaining 5 pre-existing non-censused \
         uses stay frozen at the P0 baseline for the Phase 2 lint audit",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-ir/src/host.rs",
        8,
        "one effect-kind extraction fallback removed by the Phase 2 typed \
         decoder; the remaining 8 pre-existing non-censused uses stay frozen",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-ir/src/eval.rs",
        3,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-c/src/memory.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-c/src/lib.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-c/src/emit.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-hip/src/emit.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-hip/src/memory.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-backend-metal/src/emit.rs",
        1,
        "pre-existing at the P0 baseline; not censused as substituting",
    ),
    // -- unwrap-or-prim: census rows 13 and 14 converted at Phase 1
    // (lower_cast and lower_transcendental raise; named_axis errors) ----
    (
        Pat::UnwrapOrPrim,
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        2,
        "to_tensor literal-precision defaults: float literals default to F32 \
         per spec (int/bool elements override); audited, not a substitution",
    ),
    // -- elemkind-wildcard-arm: ZERO entries left - census row 5's
    // wildcard deleted at Phase 1 (section C4.1); elem_kind is an
    // exhaustive Result-returning match ---------------------------------
    // -- unwrap-or-numeric-literal (added at Phase 1 per the chelis#776
    // census-maintenance obligation; every entry is annotated
    // proven-structural or P1-frozen; the expand-axis and tuple-get
    // index sites chelis#782 flagged were CONVERTED, not baselined) -----
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-ir/src/lower.rs",
        13,
        "structural at the P1 baseline: recursion-depth counter, \
         desync-guarded rank/extent reads, the uniform-ConstTensor \
         first-element read (non-empty by the windows(2) guard). FLAGGED, \
         not proven: conv2d's present-but-non-literal stride unwrap_or(1) \
         / padding unwrap_or(0) - the chelis#776 shape (census row 23) - \
         and the with-seed defaults, whose effects-checker cover the \
         chelis#793 red team pierced (a negative .dp int64 seed extracts \
         to None and falls to seed 0) - that .dp repro is now rejected at \
         CHECK time by chelis#793's negative-seed checker case, so the \
         sites are checker-guarded pending their census rows",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-ir/src/dag.rs",
        1,
        "proven-structural at the P1 baseline (symbolic-dim bookkeeping)",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-ir/src/eval.rs",
        2,
        "P1-frozen (shrink extent reads); Phase 2 lint audits them",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-backend-c/src/host_emit.rs",
        1,
        "proven-structural: tensor-helper root count floor (max(1))",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-backend-hip/src/emit.rs",
        2,
        "proven-structural: fused-input count over an empty set; MAX_DIM \
         zero-padding of pinned pad/shrink offset vectors",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-backend-metal/src/emit.rs",
        1,
        "proven-structural: MAX_DIM zero-padding of movement dim vectors",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-compiler-api/src/context.rs",
        1,
        "P1-frozen; Phase 2 lint audits it",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-compiler-api/src/cache_envelope.rs",
        1,
        "P1-frozen (cache bookkeeping); Phase 2 lint audits it",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        3,
        "with-seed default (mirrors lower.rs; the chelis#793 negative-seed \
         repro is now checker-rejected) and a scalarization first-element \
         read; plus the `map_or(-1_i64, ...)` process-exit-code default \
         (proven-structural: a signal-killed child has no exit code, and -1 \
         is the conventional sentinel, not a chelis#703 value substitution) \
         newly counted by the rt791 F6 widening; P1-frozen for the Phase 2 \
         lint audit",
    ),
    (
        Pat::UnwrapOrNumericLiteral,
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        1,
        "proven-structural: absent symbolic-dim label defaults to extent 1 \
         per the summary contract",
    ),
    // -- c-format-narrowing (hosted for the chelis#732 plan's Phase 0; its
    // production allowlist only ever shrinks - deleted by #732 Phase 2) ----
    (
        Pat::CFormatNarrowing,
        "crates/chelis-backend-c/src/host_emit.rs",
        4,
        "chelis#732 production allowlist: the emitted tensor print helper's \
         %.1f/%.16g split and the two scalar %.16g print sites \
         (chelis#716/#723); deleted by #732 Phase 2's generated helper",
    ),
    (
        Pat::CFormatNarrowing,
        "crates/chelis-runtime/src/lib.rs",
        1,
        "chelis#732 production allowlist: tensor_to_string's {value:.1} \
         near-integer arm; deleted by #732 Phase 2",
    ),
    (
        Pat::CFormatNarrowing,
        "crates/chelis-backend-c/src/lib.rs",
        5,
        "benign per the #732 handoff: printf tokens inside cfg(test) \
         hand-written main.c fixture strings, not product exits",
    ),
];

/// Count the numeric-literal-default spellings on one source line: the
/// direct `.unwrap_or(<lit>)`, the closure `.unwrap_or_else(|| <lit>)`, and
/// the `.map_or(<lit>, ...)` form. rt791 F6 (chelis#730 Phase 2) added the
/// latter two: `.unwrap_or_else(|| 0)` / `.map_or(0, ...)` are the same
/// silent numeric default the direct spelling produces, dressed in a
/// closure to slip past the original single-token scan. A non-numeric
/// default (`.unwrap_or_else(|| compute())`, `.map_or(other, ...)`) is not
/// counted - only a baked numeric literal is the chelis#703 shape.
fn count_numeric_defaults(line: &str) -> usize {
    // Directly after the delimiter (no whitespace trim) - preserves the
    // exact pre-widening `.unwrap_or(` count so existing baselines do not
    // shift.
    fn immediately_numeric(s: &str) -> bool {
        let s = s.strip_prefix('-').unwrap_or(s);
        s.starts_with(|c: char| c.is_ascii_digit())
    }
    // After the closure head `||`, a space is idiomatic (`|| 0`), so the
    // widened spellings trim leading whitespace before the literal check.
    fn numeric_after_ws(s: &str) -> bool {
        immediately_numeric(s.trim_start())
    }
    fn count(line: &str, needle: &str, check: fn(&str) -> bool) -> usize {
        let mut hits = 0usize;
        let mut rest = line;
        while let Some(pos) = rest.find(needle) {
            let after = &rest[pos + needle.len()..];
            if check(after) {
                hits += 1;
            }
            rest = after;
        }
        hits
    }
    // `.unwrap_or(<lit>)` (NOT `.unwrap_or_else(...)`: its substring is
    // `.unwrap_or_` so `.unwrap_or(` never matches it), plus the two
    // closure/map spellings rt791 F6 added.
    count(line, ".unwrap_or(", immediately_numeric)
        + count(line, ".unwrap_or_else(||", numeric_after_ws)
        + count(line, ".map_or(", numeric_after_ws)
}

/// Recursively collect `.rs` files under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// All `.rs` files under `crates/<crate>/src/`, as repo-relative
/// `/`-separated paths paired with contents.
fn crate_sources(root: &Path) -> Vec<(String, String)> {
    let crates_dir = root.join("crates");
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(&crates_dir) else {
        return files;
    };
    for entry in entries.flatten() {
        let src = entry.path().join("src");
        if !src.is_dir() {
            continue;
        }
        let mut paths = Vec::new();
        collect_rs_files(&src, &mut paths);
        for path in paths {
            let rel = path
                .strip_prefix(root)
                .expect("collected under root")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            let content = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
            files.push((rel, content));
        }
    }
    files
}

/// Scan the tree and diff against a baseline. Returns human-readable
/// violations; empty means the tree is at the frozen baseline.
fn violations(root: &Path, baseline: &[Entry]) -> Vec<String> {
    let sources = crate_sources(root);
    let mut found: BTreeMap<(Pat, String), usize> = BTreeMap::new();
    for (rel, content) in &sources {
        for &pat in ALL_PATS {
            if !pat.scopes().iter().any(|scope| rel.starts_with(scope)) {
                continue;
            }
            let n = pat.count(content);
            if n > 0 {
                found.insert((pat, rel.clone()), n);
            }
        }
    }

    let mut out = Vec::new();
    for ((pat, rel), n) in &found {
        let allowed = baseline
            .iter()
            .find(|(bpat, bpath, _, _)| bpat == pat && *bpath == rel.as_str())
            .map(|(_, _, count, _)| *count)
            .unwrap_or(0);
        if *n > allowed {
            out.push(format!(
                "NEW recidivist token `{}` in {rel}: found {n}, baseline {allowed}. \
                 A new site violates {}. Either remove it, or treat it as new \
                 work: file an issue, record it in the owning doc, and extend \
                 BASELINE in crates/chelis-cli/tests/loud_unsupported_tripwire.rs \
                 in the same change set.",
                pat.id(),
                pat.doc()
            ));
        }
    }
    for (pat, rel, allowed, note) in baseline {
        let n = found.get(&(*pat, (*rel).to_string())).copied().unwrap_or(0);
        if n < *allowed {
            out.push(format!(
                "token `{}` in {rel} dropped below baseline: found {n}, baseline \
                 {allowed} ({note}). If the site was fixed, shrink BASELINE in the \
                 fixing PR (spec/design/loud_unsupported.md B1: census removals \
                 only with the site's fix).",
                pat.id()
            ));
        }
    }
    out
}

fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/chelis-cli has a workspace root two levels up")
        .to_path_buf();
    assert!(
        root.join("crates").is_dir() && root.join("spec").is_dir(),
        "workspace root discovery failed: {root:?}"
    );
    root
}

/// The tripwire proper: the tree carries exactly the frozen baseline of
/// recidivist tokens - no new silent-substitution site, and no unrecorded
/// fix either.
#[test]
fn source_tree_is_at_the_frozen_substitution_baseline() {
    let got = violations(&repo_root(), BASELINE);
    assert!(
        got.is_empty(),
        "loud-unsupported tripwire (spec/design/loud_unsupported.md \
         section C4.3):\n{}",
        got.join("\n")
    );
}

/// The tripwire's own negative test (the Phase 0 oracle): a planted
/// `/* unsupported ... */ 0` fallback must go red.
#[test]
fn tripwire_goes_red_on_a_planted_stub_fallback() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("crates/planted-backend/src");
    fs::create_dir_all(&src).expect("mkdir");
    fs::write(
        src.join("emit.rs"),
        "fn emit(other: &str) -> String {\n    \
         format!(\"/* unsupported builtin {other} */ 0\")\n}\n",
    )
    .expect("write planted file");
    let got = violations(dir.path(), &[]);
    assert!(
        got.iter()
            .any(|v| v.contains("silent-stub-zero") && v.contains("planted-backend")),
        "the tripwire must flag a planted `*/ 0` fallback; got: {got:?}"
    );
}

/// Every token class trips, not just the stub-zero one, and a clean tree
/// scans clean.
#[test]
fn tripwire_goes_red_on_every_planted_token_class() {
    let dir = tempfile::tempdir().expect("tempdir");
    let clean = dir.path().join("crates/clean-crate/src");
    fs::create_dir_all(&clean).expect("mkdir");
    fs::write(clean.join("lib.rs"), "pub fn ok() {}\n").expect("write");
    assert!(
        violations(dir.path(), &[]).is_empty(),
        "a clean tree must scan clean"
    );
    // A COMMENT mentioning the numeric-unwrap token must not count as a
    // live site (conversion comments quote the pre-fix code).
    let commented = dir.path().join("crates/chelis-ir/src");
    fs::create_dir_all(&commented).expect("mkdir");
    fs::write(
        commented.join("commented.rs"),
        "// the pre-fix code read bound.unwrap_or(0.0) here\npub fn ok() {}\n",
    )
    .expect("write");
    assert!(
        violations(dir.path(), &[])
            .iter()
            .all(|v| !v.contains("unwrap-or-numeric-literal")),
        "a comment-only token mention must not trip the numeric-unwrap class"
    );
    fs::remove_file(commented.join("commented.rs")).expect("cleanup");

    let ir = dir.path().join("crates/chelis-ir/src");
    fs::create_dir_all(&ir).expect("mkdir");
    fs::write(
        ir.join("lower.rs"),
        "fn f() {\n    let w = windows.unwrap_or_default();\n    \
         let p = prec.unwrap_or(Prim::F32);\n    \
         let low = bound.unwrap_or(0.0);\n}\n",
    )
    .expect("write");
    let hip = dir.path().join("crates/chelis-backend-hip/src");
    fs::create_dir_all(&hip).expect("mkdir");
    fs::write(
        hip.join("emit.rs"),
        "fn k(p: Prim) -> ElemKind {\n    match p {\n        \
         _ => kernels::ElemKind::F32,\n    }\n}\n",
    )
    .expect("write");
    let c = dir.path().join("crates/chelis-backend-c/src");
    fs::create_dir_all(&c).expect("mkdir");
    fs::write(
        c.join("host_emit.rs"),
        "fn p() -> String {\n    let s = \"printf(\\\"%.16g\\\", v);\";\n    \
         let t = \"chelis_string_from_cstr(\\\"<value>\\\")\";\n    \
         format!(\"/* unsupported builtin x */ 0\")\n}\n",
    )
    .expect("write");

    let rt = dir.path().join("crates/chelis-runtime/src");
    fs::create_dir_all(&rt).expect("mkdir");
    fs::write(
        rt.join("lib.rs"),
        "fn s(value: f64) -> String {\n    format!(\"{value:.1}\")\n}\n",
    )
    .expect("write");

    let got = violations(dir.path(), &[]);
    for id in [
        "silent-stub-zero",
        "value-placeholder",
        "unwrap-or-default",
        "unwrap-or-prim",
        "elemkind-wildcard-arm",
        "unwrap-or-numeric-literal",
        "c-format-narrowing",
    ] {
        assert!(
            got.iter().any(|v| v.contains(id)),
            "planted token class `{id}` must trip; got: {got:?}"
        );
    }
    assert!(
        got.iter().any(|v| v.contains("chelis-runtime")
            && v.contains("c-format-narrowing")
            && v.contains("faithful_observation.md")),
        "the planted {{value:.1}} narrowing arm must trip with the chelis#732 \
         doc pointer; got: {got:?}"
    );
}

/// rt791 F6 (chelis#730 Phase 2): the closure/`map_or` numeric-default
/// spellings that evaded the original single-token scan now join the
/// `unwrap-or-numeric-literal` class. Positive: each spelling counts.
#[test]
fn count_numeric_defaults_counts_all_three_spellings() {
    assert_eq!(count_numeric_defaults("let x = a.unwrap_or(0);"), 1);
    assert_eq!(count_numeric_defaults("let x = a.unwrap_or(-1);"), 1);
    assert_eq!(count_numeric_defaults("let x = a.unwrap_or_else(|| 0);"), 1);
    assert_eq!(
        count_numeric_defaults("let x = a.unwrap_or_else(|| -1_i64);"),
        1
    );
    assert_eq!(count_numeric_defaults("let x = a.map_or(0, f);"), 1);
    assert_eq!(count_numeric_defaults("let x = a.map_or(-1_i64, f);"), 1);
    // Multiple spellings on one line all count.
    assert_eq!(
        count_numeric_defaults("a.unwrap_or(0) + b.unwrap_or_else(|| 1) + c.map_or(2, f)"),
        3
    );
}

/// Negative parity: a NON-numeric default is not the chelis#703 shape and
/// must not count, and `.unwrap_or_else(...)` must not be double-counted by
/// the `.unwrap_or(` scan.
#[test]
fn count_numeric_defaults_ignores_non_numeric_defaults() {
    assert_eq!(
        count_numeric_defaults("let x = a.unwrap_or(default_val);"),
        0
    );
    assert_eq!(
        count_numeric_defaults("let x = a.unwrap_or_else(|| compute());"),
        0
    );
    assert_eq!(
        count_numeric_defaults("let x = a.unwrap_or_else(|| Prim::F32);"),
        0
    );
    assert_eq!(count_numeric_defaults("let x = a.map_or(other, f);"), 0);
    // `unwrap_or_else` with a numeric literal counts exactly once (the
    // closure scan), never also via the `.unwrap_or(` scan.
    assert_eq!(count_numeric_defaults("let x = a.unwrap_or_else(|| 0);"), 1);
    // A bare closure with no default and an unrelated numeric are inert.
    assert_eq!(count_numeric_defaults("let n = 0; let f = || 0;"), 0);
}

/// The widening trips inside a planted lowering file for the two new
/// spellings, and a non-numeric closure default stays clean.
#[test]
fn tripwire_widening_trips_on_planted_closure_defaults() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ir = dir.path().join("crates/chelis-ir/src");
    fs::create_dir_all(&ir).expect("mkdir");
    fs::write(
        ir.join("lower.rs"),
        "fn f() {\n    let a = seed.unwrap_or_else(|| 0);\n    \
         let b = code.map_or(-1, i64::from);\n}\n",
    )
    .expect("write");
    let got = violations(dir.path(), &[]);
    assert!(
        got.iter().any(|v| v.contains("unwrap-or-numeric-literal")),
        "planted closure/map_or numeric defaults must trip; got: {got:?}"
    );

    // Negative: a non-numeric closure default in the same scope is clean.
    fs::write(
        ir.join("lower.rs"),
        "fn f() {\n    let a = seed.unwrap_or_else(|| compute());\n    \
         let b = code.map_or(fallback, i64::from);\n}\n",
    )
    .expect("write");
    assert!(
        !violations(dir.path(), &[])
            .iter()
            .any(|v| v.contains("unwrap-or-numeric-literal")),
        "non-numeric closure/map_or defaults must NOT trip"
    );
}
