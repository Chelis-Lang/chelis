//! chelis#730 Phase 0 - the token tripwire (spec/design/loud_unsupported.md
//! section C4.3).
//!
//! A fast source-tree grep over the recidivist tokens of the chelis#703
//! class (unsupported cases silently substitute a value). The verified
//! section C5 census is frozen here as `BASELINE`; any NEW occurrence of a
//! recidivist token goes red with a pointer back at the plan. This exists
//! purely to bridge until the section C4.2 lint rule and the section C3
//! Result channel land (Phase 1-2), and to catch generated-string contexts
//! the lint cannot see.
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
//!   fixing PR (removals only with the site's fix).

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
            Pat::UnwrapOrPrim => LOWERING_EMISSION_AND_RUNTIME_PACKING,
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
    // -- silent-stub-zero ---------------------------------------------------
    (
        Pat::StubZero,
        "crates/chelis-backend-c/src/host_emit.rs",
        1,
        "census row 2 (chelis#682/#704/#705/#715): the builtin literal-0 stub",
    ),
    (
        Pat::StubZero,
        "crates/chelis-backend-metal/src/emit.rs",
        1,
        "census row 19 (chelis#745): Metal host_scalar_literal catch-all, \
         found by this tripwire's Phase 0 sweep; dead behind the f64 gate",
    ),
    // -- value-placeholder --------------------------------------------------
    (
        Pat::ValuePlaceholder,
        "crates/chelis-backend-c/src/host_emit.rs",
        3,
        "census rows 3 (chelis#734, to_string catch-all) and 4 (chelis#714 \
         symptom, the two unclassifiable-print sites)",
    ),
    // -- unwrap-or-default --------------------------------------------------
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-ir/src/lower.rs",
        7,
        "census row 8 (chelis#725, reduce_window extraction, 2 of these) + 5 \
         pre-existing non-censused uses frozen at the P0 baseline",
    ),
    (
        Pat::UnwrapOrDefault,
        "crates/chelis-ir/src/host.rs",
        9,
        "pre-existing at the P0 baseline; not censused as substituting; \
         Phase 2 lint audits them",
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
    // -- unwrap-or-prim -----------------------------------------------------
    (
        Pat::UnwrapOrPrim,
        "crates/chelis-ir/src/lower.rs",
        2,
        "census row 13 (lower_transcendental precision default at :9833; \
         lower_cast fallback at :10235, LIVE via the .dp build lane, \
         chelis#744); section C1.4 applies at Phase 1",
    ),
    (
        Pat::UnwrapOrPrim,
        "crates/chelis-compiler-api/src/runtime/named_axis.rs",
        1,
        "census row 14 (pack_dag_roots root-precision default); dead by \
         canary, section C1.4 applies at Phase 1",
    ),
    (
        Pat::UnwrapOrPrim,
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        2,
        "to_tensor literal-precision defaults: float literals default to F32 \
         per spec (int/bool elements override); audited, not a substitution",
    ),
    // -- elemkind-wildcard-arm ----------------------------------------------
    (
        Pat::ElemKindWildcardArm,
        "crates/chelis-backend-hip/src/emit.rs",
        1,
        "census row 5 (chelis#689): elem_kind's F32 fallback; deleted at \
         Phase 1 (section C4.1)",
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

    let ir = dir.path().join("crates/chelis-ir/src");
    fs::create_dir_all(&ir).expect("mkdir");
    fs::write(
        ir.join("lower.rs"),
        "fn f() {\n    let w = windows.unwrap_or_default();\n    \
         let p = prec.unwrap_or(Prim::F32);\n}\n",
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
