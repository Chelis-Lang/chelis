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
//! Also hosted here on behalf of the chelis#732 plan (Phase 0 item 3:
//! "coordinate, do not duplicate", per the roadmap's Wave 0 handshake;
//! widened to the Rust lane per its B2.4/B2.8 instrument list): the three
//! no-third-formatter classes - `c-format-narrowing` (C printf tokens),
//! `rust-format-narrowing` (Rust precision-spec forms, all crate src), and
//! `rust-debug-numeric-format` (Debug-format tokens at the declared
//! `OBSERVATION_EXIT_SURFACES`, directory-prefixed so a NEW file inside a
//! declared surface is born covered at baseline zero). New-site violations
//! for all three point at `spec/design/faithful_observation.md` B2.4. The
//! C class's production allowlist is empty since #732 Phase 2 and stays
//! that way; the Rust classes' baselines are annotated non-exit carriers.
//! The oracle side (`scripts/faithful_observation_phase2_oracle.py`)
//! cross-checks every class's baseline paths against its own permitted
//! sets and requires every Pat whose doc() cites faithful_observation.md
//! to carry an oracle coverage row - a hosted class it cannot see is a
//! structural failure there, not a quiet gap here.
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
//! Known limits of this source inventory (PR #746 review; extended with
//! the Rust-lane classes):
//! - FALSE-RED surface: the format classes scan for very common tokens
//!   (`%.1f`/`%.16g` everywhere; `{x:.N}` everywhere; `:?}` at the
//!   declared exit surfaces, where diagnostics legitimately Debug-print
//!   mismatched values). A NEW legitimate use (a timing print, a
//!   benchmark, a diagnostic) trips the increase branch; if it is
//!   genuinely not a numeric-observation exit, extend `BASELINE` with an
//!   annotation in the same PR - the annotated-bump protocol. The
//!   faithful_observation.md B2.4 pointer in the message applies only to
//!   formatter sites.
//! - FALSE-GREEN evasions (grep is textual): `*/ 0.0"` / `*/0"` /
//!   unquoted stubs evade `StubZero`; a multi-line `_ =>` arm evades
//!   `ElemKindWildcardArm` (same-line match only), as do NAMED catch-alls
//!   (`other =>`) - though a named catch-all emitting a stub is caught by
//!   `StubZero`, per census row 19; non-quote-adjacent `<value>`
//!   spellings evade `ValuePlaceholder`. For the no-third-formatter
//!   classes the DECLARED residue (faithful_observation.md B2.4, each
//!   piece with an owner): derived-Debug containers embedding floats
//!   (`{other:?}` on a `#[derive(Debug)]` value - chelis#729's payload
//!   work plus the review rule), bare `{}` Display / `.to_string()` of a
//!   numeric payload (review rule), and exits created outside
//!   `OBSERVATION_EXIT_SURFACES` (review rule; a new exit surface adds
//!   its prefix in the same change set). The typed boundaries and their
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
    /// `%.16g` / `%.1f` anywhere in crate sources - the C-lane f64-shaped
    /// observation-channel exits (chelis#716/#723/#728; owned by the
    /// chelis#732 plan, hosted in this tripwire per its Phase 0 item 3
    /// and the Wave 0 handshake; a new site violates
    /// spec/design/faithful_observation.md B2.4, no third formatter).
    /// The Rust `{value:.1}` spelling moved to `RustFormatNarrowing`,
    /// whose precision-spec scan subsumes it; this class is purely C
    /// format tokens, matching its name.
    CFormatNarrowing,
    /// A Rust precision-format spec (`{v:.17}`, `{:.8}`, `{rel_err:.2e}`)
    /// on a non-comment line, anywhere in crate sources - the Rust-lane
    /// twin of `CFormatNarrowing`: a fixed decimal precision applied to a
    /// numeric payload narrows or pads it away from the shortest
    /// round-trip grammar, the same third-formatter shape in the other
    /// lane (chelis#732 B2.4; added by the 2026-07-30 detector-scope
    /// review, which found the rule's instrument saw only C tokens).
    RustFormatNarrowing,
    /// A Debug-format token (`:?}` or `:#?}`) on a non-comment line at
    /// the declared `OBSERVATION_EXIT_SURFACES` - the exact spelling of
    /// PR #891's `format_f64_json` (`format!("{v:?}")`), a second Rust
    /// implementation of the normative grammar beside `format_element`.
    /// Byte-identical today is not a defense: a second implementation
    /// makes the one-change-set migration protocol unhonorable
    /// (chelis#732 B2.4). The scope is directory-prefixed so a NEW file
    /// inside a declared exit surface starts at baseline zero.
    RustDebugNumericFormat,
}

const ALL_PATS: &[Pat] = &[
    Pat::StubZero,
    Pat::ValuePlaceholder,
    Pat::UnwrapOrDefault,
    Pat::UnwrapOrPrim,
    Pat::ElemKindWildcardArm,
    Pat::UnwrapOrNumericLiteral,
    Pat::CFormatNarrowing,
    Pat::RustFormatNarrowing,
    Pat::RustDebugNumericFormat,
];

/// Every `Pat` has a stable index, and the `match` forces this function to
/// grow with the enum: a variant added without updating it is a compile
/// error, and `all_pats_lists_every_variant_exactly_once` then forces the
/// hand-maintained `ALL_PATS` list to carry it - a detector that exists
/// but never scans is unrepresentable (faithful_observation.md B2.8).
fn variant_index(pat: Pat) -> usize {
    match pat {
        Pat::StubZero => 0,
        Pat::ValuePlaceholder => 1,
        Pat::UnwrapOrDefault => 2,
        Pat::UnwrapOrPrim => 3,
        Pat::ElemKindWildcardArm => 4,
        Pat::UnwrapOrNumericLiteral => 5,
        Pat::CFormatNarrowing => 6,
        Pat::RustFormatNarrowing => 7,
        Pat::RustDebugNumericFormat => 8,
    }
}

const PAT_VARIANT_COUNT: usize = 9;

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
            Pat::RustFormatNarrowing => "rust-format-narrowing",
            Pat::RustDebugNumericFormat => "rust-debug-numeric-format",
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
        // The chelis#732 census's Rust exit surfaces: where numeric
        // payloads become user-visible text. DIRECTORY prefixes on
        // purpose - a new file inside a declared surface (the PR #891
        // `runtime/json.rs` shape) is covered from its first line with an
        // implicit baseline of zero. `chelis-types` is file-scoped to the
        // sanctioned formatter module (a directory scope would drag in
        // infer.rs's dozens of non-exit diagnostics for no exit
        // coverage). Adding a NEW exit surface anywhere else obliges
        // adding its prefix here in the same change set
        // (faithful_observation.md B2.4).
        const OBSERVATION_EXIT_SURFACES: &[&str] = &[
            "crates/chelis-types/src/observation.rs",
            "crates/chelis-runtime/src/",
            "crates/chelis-compiler-api/src/runtime/",
        ];
        match self {
            Pat::StubZero
            | Pat::ValuePlaceholder
            | Pat::CFormatNarrowing
            | Pat::RustFormatNarrowing => ALL_CRATE_SRC,
            Pat::UnwrapOrDefault => LOWERING_AND_EMISSION,
            Pat::UnwrapOrPrim | Pat::UnwrapOrNumericLiteral => {
                LOWERING_EMISSION_AND_RUNTIME_PACKING
            }
            Pat::ElemKindWildcardArm => BACKENDS,
            Pat::RustDebugNumericFormat => OBSERVATION_EXIT_SURFACES,
        }
    }

    /// The design doc a NEW occurrence of this pattern violates. The
    /// Phase 2 oracle parses this function: every arm citing
    /// faithful_observation.md must have an oracle coverage row there
    /// (its doc-citation parity leg), so keep the citation literal.
    fn doc(self) -> &'static str {
        match self {
            Pat::CFormatNarrowing | Pat::RustFormatNarrowing | Pat::RustDebugNumericFormat => {
                "spec/design/faithful_observation.md B2.4 (no third formatter)"
            }
            _ => "spec/design/loud_unsupported.md B2.5",
        }
    }

    /// The maintenance rule a count DECREASE points at: an unrecorded fix
    /// narrows what the baseline claims to cover, so the shrink must ride
    /// the fixing PR under the owning doc's protocol.
    fn maintenance(self) -> &'static str {
        match self {
            Pat::CFormatNarrowing | Pat::RustFormatNarrowing | Pat::RustDebugNumericFormat => {
                "spec/design/faithful_observation.md B2.9: baselines are \
                 declared boundaries; shrink this one in the fixing PR"
            }
            _ => {
                "spec/design/loud_unsupported.md B1: census removals \
                 only with the site's fix"
            }
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
            Pat::CFormatNarrowing => occurrences(content, "%.16g") + occurrences(content, "%.1f"),
            // Comment lines are skipped for both Rust-lane classes so
            // prose ABOUT the grammar (observation.rs's doc comments
            // quote `{:?}` as the normative spelling) does not count as a
            // live formatting site.
            Pat::RustFormatNarrowing => content
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .map(count_precision_format_specs)
                .sum(),
            Pat::RustDebugNumericFormat => content
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .map(count_debug_format_tokens)
                .sum(),
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
        6,
        "one effect-kind extraction fallback removed by the Phase 2 typed \
         decoder and one generic-ADT substitution default removed by the \
         applied-type Result boundary; source-reconstructed ADT parameters \
         now come from the checker registry; the remaining 6 pre-existing \
         non-censused uses stay frozen",
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
    // -- c-format-narrowing (hosted for the chelis#732 plan's Phase 0) ----
    // The production allowlist rows (host_emit.rs = 4, chelis-runtime
    // lib.rs = 1) were DELETED by chelis#732 Phase 2 as the Phase 0
    // handoff promised: every compiled-lane float exit now routes through
    // `chelis_format_shortest` / the generated print helper, so the
    // format-narrowing token count in product code is zero and any new
    // occurrence is a third formatter (faithful_observation.md section
    // B2.4, a review-blocking finding).
    (
        Pat::CFormatNarrowing,
        "crates/chelis-backend-c/src/lib.rs",
        5,
        "benign per the #732 handoff: printf tokens inside cfg(test) \
         hand-written main.c fixture strings, not product exits",
    ),
    // -- rust-format-narrowing (hosted for chelis#732 B2.4's Rust lane,
    // added by the 2026-07-30 detector-scope review). Every row is an
    // annotated NON-EXIT carrier: no production numeric-observation exit
    // uses a precision spec today, and a new one is a third formatter. --
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-backend-c/src/lib.rs",
        4,
        "cfg(test): the fill negative-lock needle ({value:.8}f must NOT \
         appear), its escaped {{:.8}} message quote, and two rel_err \
         tolerance-assert messages - not product exits",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-backend-hip/src/emit.rs",
        2,
        "cfg(test) negative-lock messages quoting the banned {{:.8}}f \
         lossy-literal form - locks AGAINST narrowing, not exits",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-cli/src/main.rs",
        2,
        "MiB/KiB memory-usage display ({:.2} of a byte count) - a UX \
         quantity, not a stored numeric payload",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-cove/src/live.rs",
        2,
        "fitness-score UI strings ({score:.2}) - cove's live display, \
         not an observation exit",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-e2e/src/bench.rs",
        2,
        "bench verdict notes (accuracy gap, tolerance target) - report \
         prose, not stored-value rendering",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-e2e/src/bin/train_mnist.rs",
        5,
        "training progress prints (loss/accuracy at {:.4}) - a manual \
         driver's console output, not a language exit",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-ir/src/grad.rs",
        2,
        "cfg(test) tolerance-assert message ({expected:.6}/{a:.6}) - not \
         an exit",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-prove/src/bin/certify_erf_envelope.rs",
        3,
        "envelope-certification eprintln reports (eps at {:.6e}) - prover \
         tooling output, not a language exit",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-prove/src/bin/certify_special_fn_envelope.rs",
        7,
        "envelope-certification eprintln reports - prover tooling output, \
         not a language exit",
    ),
    (
        Pat::RustFormatNarrowing,
        "crates/chelis-prove/src/opaque.rs",
        3,
        "fuzz-acceptance-rate diagnostic ({:.4} ratios) - prover \
         diagnostics, not a stored-value exit",
    ),
    // -- rust-debug-numeric-format (hosted for chelis#732 B2.4's Rust
    // lane; scoped to OBSERVATION_EXIT_SURFACES). The first two rows ARE
    // the sanctioned formatters - `{:?}` is the normative grammar's own
    // definition there. The compiler-api rows are the DECLARED
    // derived-Debug residue carriers per faithful_observation.md B2.4:
    // diagnostics that Debug-print mismatched values through derive(Debug),
    // owned by chelis#729's payload work plus the review rule. -----------
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-types/src/observation.rs",
        27,
        "the sanctioned formatter itself: format_element's F32/F64 arms \
         (the normative grammar IS Rust {:?}) plus cfg(test) grammar \
         expectations",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-runtime/src/format_shortest.rs",
        2,
        "the sanctioned compiled-lane routine: chelis_format_shortest's \
         F64/F32 arms, byte-locked against format_element",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        50,
        "declared derived-Debug residue carriers: Err(format!) \
         type-mismatch diagnostics over Value/Prim shapes, the tensor \
         SHAPE debug in render_tensor (elements route through \
         format_element), and cfg(test) assertions",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        24,
        "declared derived-Debug residue carriers: Err(format!) \
         diagnostics over Value/callable/handle shapes",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/invariant.rs",
        1,
        "non-boolean-predicate diagnostic ({other:?}); numeric payloads \
         in this file route through format_element",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/named_axis.rs",
        1,
        "axis-mismatch diagnostic - a declared derived-Debug residue \
         carrier",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/tests.rs",
        20,
        "cfg-gated runtime unit-test assertions, not product exits",
    ),
    (
        Pat::RustDebugNumericFormat,
        "crates/chelis-compiler-api/src/runtime/transforms.rs",
        1,
        "transform-shape diagnostic - a declared derived-Debug residue \
         carrier",
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

/// Count Rust precision-format specs on one source line: `{v:.17}`,
/// `{:.8}`, `{rel_err:.2e}`, `{dur:.3?}` - an interpolation whose format
/// spec pins a decimal precision. The shape matched is `{`, an optional
/// `[A-Za-z0-9_]*` argument, `:.`, one or more ASCII digits, optionally
/// ONE trailing format-type character (a letter such as `e`, or `?`),
/// then `}`. Escaped `{{:.8}}` doc-quotes in assertion messages match too
/// (the walk-back sees the inner brace); that is deliberate conservatism,
/// absorbed by an annotated baseline row rather than a scanner exception.
fn count_precision_format_specs(line: &str) -> usize {
    let bytes = line.as_bytes();
    let mut hits = 0usize;
    let mut from = 0usize;
    while let Some(pos) = line[from..].find(":.") {
        let colon = from + pos;
        let mut j = colon + 2;
        let digits_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        let mut shape_ok = j > digits_start;
        if shape_ok && j < bytes.len() && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'?') {
            j += 1;
        }
        shape_ok = shape_ok && j < bytes.len() && bytes[j] == b'}';
        if shape_ok {
            let mut k = colon;
            while k > 0 && (bytes[k - 1].is_ascii_alphanumeric() || bytes[k - 1] == b'_') {
                k -= 1;
            }
            if k > 0 && bytes[k - 1] == b'{' {
                hits += 1;
            }
        }
        from = colon + 2;
    }
    hits
}

/// Count Debug-format tokens on one source line: the plain `:?}` and the
/// pretty `:#?}` (whose bytes do not contain the plain token, so the two
/// counts never overlap).
fn count_debug_format_tokens(line: &str) -> usize {
    line.matches(":?}").count() + line.matches(":#?}").count()
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
                 fixing PR ({}).",
                pat.id(),
                pat.maintenance()
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
        "fn s(value: f64) -> String {\n    format!(\"{value:.1}\")\n}\n\
         fn d(v: f64) -> String {\n    format!(\"{v:?}\")\n}\n",
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
        "rust-format-narrowing",
        "rust-debug-numeric-format",
    ] {
        assert!(
            got.iter().any(|v| v.contains(id)),
            "planted token class `{id}` must trip; got: {got:?}"
        );
    }
    assert!(
        got.iter().any(|v| v.contains("chelis-runtime")
            && v.contains("rust-format-narrowing")
            && v.contains("faithful_observation.md")),
        "the planted {{value:.1}} narrowing arm must trip the Rust precision \
         class with the chelis#732 doc pointer; got: {got:?}"
    );
    assert!(
        got.iter().any(|v| v.contains("chelis-runtime")
            && v.contains("rust-debug-numeric-format")
            && v.contains("faithful_observation.md")),
        "the planted {{v:?}} arm must trip the Debug class with the \
         chelis#732 doc pointer; got: {got:?}"
    );
}

/// The B2.8 exhaustiveness guard's executable half: `variant_index`'s
/// `match` forces itself to grow with the enum at compile time, and this
/// test forces `ALL_PATS` to carry every variant exactly once - so a
/// detector that exists but never scans cannot be written.
#[test]
fn all_pats_lists_every_variant_exactly_once() {
    let mut seen = [0usize; PAT_VARIANT_COUNT];
    for &pat in ALL_PATS {
        seen[variant_index(pat)] += 1;
    }
    assert!(
        seen.iter().all(|&n| n == 1),
        "ALL_PATS must list every Pat variant exactly once; \
         occurrence vector by variant_index: {seen:?}"
    );
}

/// The precision-spec scanner: every narrowing spelling counts, on any
/// argument form, and the format-type suffix is covered.
#[test]
fn count_precision_format_specs_counts_narrowing_spellings() {
    assert_eq!(count_precision_format_specs("format!(\"{v:.17}\")"), 1);
    assert_eq!(count_precision_format_specs("format!(\"{:.8}\", x)"), 1);
    assert_eq!(
        count_precision_format_specs("format!(\"{rel_err:.2e}\")"),
        1
    );
    assert_eq!(count_precision_format_specs("format!(\"{value:.1}\")"), 1);
    assert_eq!(count_precision_format_specs("format!(\"{dur:.3?}\")"), 1);
    assert_eq!(
        count_precision_format_specs("write!(f, \"{a:.4} vs {b:.4}\")"),
        2
    );
    // The escaped doc-quote form counts too (deliberate conservatism,
    // absorbed by an annotated baseline row).
    assert_eq!(count_precision_format_specs("\"quotes {{:.8}} form\""), 1);
}

/// Negative parity for the precision scanner: non-precision format specs
/// and non-interpolation colons stay clean.
#[test]
fn count_precision_format_specs_ignores_non_precision_forms() {
    assert_eq!(count_precision_format_specs("format!(\"{v:?}\")"), 0);
    assert_eq!(count_precision_format_specs("format!(\"{name}\")"), 0);
    assert_eq!(count_precision_format_specs("format!(\"{n:>8}\")"), 0);
    assert_eq!(count_precision_format_specs("format!(\"{n:08}\")"), 0);
    // A precision spec with no closing brace right after, or no digits,
    // is not the shape.
    assert_eq!(
        count_precision_format_specs("let r = a..b; r.contains(&x)"),
        0
    );
    assert_eq!(count_precision_format_specs("format!(\"{v:.}\")"), 0);
    // Outside an interpolation entirely (no opening brace on walk-back).
    assert_eq!(count_precision_format_specs("path :.4} nonsense"), 0);
}

/// The Debug-token scanner: plain and pretty forms count, positional and
/// named, and the two tokens never double-count one site.
#[test]
fn count_debug_format_tokens_counts_debug_forms() {
    assert_eq!(count_debug_format_tokens("format!(\"{v:?}\")"), 1);
    assert_eq!(count_debug_format_tokens("format!(\"{:?}\", other)"), 1);
    assert_eq!(count_debug_format_tokens("format!(\"{x:#?}\")"), 1);
    assert_eq!(count_debug_format_tokens("\"{a:?} then {b:#?}\""), 2);
    assert_eq!(count_debug_format_tokens("format!(\"{name}\")"), 0);
    assert_eq!(count_debug_format_tokens("format!(\"{v:.2}\")"), 0);
}

/// The chelis#732 B2.4 new-file guarantee (the PR #891 shape, executed):
/// a `format_f64_json`-style second formatter born in a NEW file inside a
/// declared exit surface trips at baseline zero, while the same content
/// outside the declared surfaces is the documented review-rule residue
/// and does not trip the Debug class.
#[test]
fn tripwire_covers_new_files_in_declared_exit_surfaces() {
    let planted = "pub fn format_f64_json(value: f64) -> String {\n    \
                   format!(\"{value:?}\")\n}\n";

    let dir = tempfile::tempdir().expect("tempdir");
    let runtime_mod = dir.path().join("crates/chelis-compiler-api/src/runtime");
    fs::create_dir_all(&runtime_mod).expect("mkdir");
    fs::write(runtime_mod.join("json.rs"), planted).expect("write");
    let got = violations(dir.path(), &[]);
    assert!(
        got.iter().any(|v| v.contains("rust-debug-numeric-format")
            && v.contains("crates/chelis-compiler-api/src/runtime/json.rs")
            && v.contains("faithful_observation.md")),
        "a new file inside a declared exit surface must trip the Debug \
         class at baseline zero; got: {got:?}"
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let rt = dir.path().join("crates/chelis-runtime/src");
    fs::create_dir_all(&rt).expect("mkdir");
    fs::write(rt.join("json.rs"), planted).expect("write");
    let got = violations(dir.path(), &[]);
    assert!(
        got.iter().any(|v| v.contains("rust-debug-numeric-format")
            && v.contains("crates/chelis-runtime/src/json.rs")),
        "a new file under the runtime crate's src must trip the Debug \
         class; got: {got:?}"
    );

    // Outside the declared surfaces: the Debug class stays silent (the
    // declared review-rule residue), and nothing else fires on this
    // content either.
    let dir = tempfile::tempdir().expect("tempdir");
    let api = dir.path().join("crates/chelis-compiler-api/src");
    fs::create_dir_all(&api).expect("mkdir");
    fs::write(api.join("context.rs"), planted).expect("write");
    let got = violations(dir.path(), &[]);
    assert!(
        !got.iter().any(|v| v.contains("rust-debug-numeric-format")),
        "content outside OBSERVATION_EXIT_SURFACES is review-rule \
         residue, not a Debug-class hit; got: {got:?}"
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
