//! Every surviving `expand` program site is accounted for, by name and count.
//!
//! chelis#1277 splits one primitive into two: `expand` keeps the same-rank
//! broadcast of a unit extent, and `insert` takes the rank-increasing form.
//! The migration renamed every rank-increasing call site. What is left must be
//! either a genuine same-rank broadcast or a test whose subject is the
//! two-candidate deferral the removal slices delete.
//!
//! This inventory is the tripwire on that claim. A new `expand` site anywhere
//! else fails it, and so does a site that disappears without its row being
//! removed, so the count cannot drift in either direction unnoticed.
//!
//! The deferral group is gone. Those files tested the choice between two
//! candidate shapes, which `spec/04-type-system.md` section 4.7.2 removed, so
//! S2b deleted or pruned them and their rows came out with them. What is left
//! is same-rank sites: each is a program whose operand carries a unit extent
//! at the axis, which is what `expand` now means.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Files that may still contain `expand` program text, with the exact count and
/// the reason the sites are there.
const ALLOWED: &[(&str, usize, &str)] = &[
    (
        "examples/checked_reshape.ch",
        1,
        "the checked C metadata example broadcasts tensor[1, 2, int64] to tensor[3, 2, int64] at its unit axis, retaining rank 2",
    ),
    // These spell the checked C metadata API, not a Chelis program call.
    // Keep exact counts so additions still require classification.
    (
        "crates/chelis-backend-c/src/emit.rs",
        1,
        "the generated C target-validation call supports the verified same-rank and inserted-axis IR forms",
    ),
    (
        "crates/chelis-backend-c/tests/checked_c_movement.rs",
        2,
        "the adoption guard and removed-validation mutant name that exact C API",
    ),
    (
        "crates/chelis-backend-c/tests/exec_compile.rs",
        1,
        "the native allocation-order control locates that exact C API call",
    ),
    (
        "scripts/dtype_phase4b_oracle.py",
        1,
        "the frozen normative registry reproduces that exact public C declaration",
    ),
    // The vendored Hull conformance corpus. Hull is the upstream type and
    // effect REFERENCE, and this corpus is a frozen snapshot pinned to Hull
    // commit 653be94e with its verdicts recorded per program. Renaming these
    // asks the compiler to typecheck a builtin Hull never verdicted, which the
    // gate reports as unexplained disagreements. The corpus is upstream's to
    // rename, not this repository's, so these 35 keep the pinned name.
    (
        "tests/conformance/hull/programs",
        35,
        "vendored upstream Hull corpus, frozen at hull_commit 653be94e",
    ),
    // A deliberate mutant: the oracle's anchor test replaces the real spec
    // line with a wrong one to prove the anchor catches it. Renaming a mutant
    // is churn that can only weaken it.
    (
        "scripts/test_dtype_phase4b_oracle.py",
        1,
        "the wrong-line mutant in the mean-adjoint anchor test",
    ),
    // Genuine same-rank broadcasts: the declared result has the operand's rank.
    // These are what `expand` means after the split, so they are not renamed.
    (
        "crates/chelis-ir/tests/lowering_trace.rs",
        1,
        "the host-boundary trace fixture broadcasts to_tensor([3.0f32]) from its unit axis to a runtime extent, retaining rank 1",
    ),
    (
        "crates/chelis-types/tests/infer_module_parity.rs",
        1,
        "broadcast_bias declares tensor[64, f32] from tensor[1, f32]",
    ),
    (
        "crates/chelis-types/tests/issue_1112_extent_dtypes.rs",
        4,
        "four rows declare tensor[8, 4, f32] or tensor[n, 4, f32] from tensor[1, 4, f32]",
    ),
    // Markdown. `spec/` is skipped above; these are the rest.
    //
    // Two `openspec/` captures still carry a rank-increasing call. They mirror
    // `spec/04-type-system.md` §4.5.3 and have been stale since chelis#1532
    // renamed it there. A capture is not authority and is not this pull
    // request's to edit: the OpenSpec migration owns re-capturing them, so
    // they are listed, not renamed.
    (
        "openspec/specs/type-system/spec.md",
        1,
        "captured capability mirroring spec/04 \u{00a7}4.5.3; the OpenSpec migration owns re-capture",
    ),
    (
        "openspec/changes/capture-type-system/specs/type-system/spec.md",
        1,
        "the same capture, in its originating change",
    ),
    // Investigation records. Each describes a program or an IR transcript as
    // it stood when it was diagnosed. Renaming the recorded text would make
    // the record disagree with the compiler output printed beside it.
    (
        "docs/investigations/implicit_copy_fanout_v3_diagnosis.md",
        2,
        "a recorded IR transcript, quoted as diagnosed",
    ),
    (
        "docs/investigations/issue_206_runtime_dim_reshape_diagnosis.md",
        1,
        "a recorded neighbour program, quoted as diagnosed",
    ),
    (
        "docs/investigations/specialize_and_backend_emit.md",
        2,
        "`emit_expand()`, a Rust backend function name in prose, not program text",
    ),
    // Gap analyses: elided sketches (`expand(a, ...)`) in tables whose subject
    // is which programs the checker admitted at the time. No rank is stated,
    // and the tables are historical.
    (
        "docs/gap_synthesis.md",
        2,
        "elided sketch in a historical admitted-programs table",
    ),
    (
        "docs/identified_gaps.md",
        2,
        "elided sketch in a historical admitted-programs table",
    ),
    // The one genuine same-rank line in the book: `expand` documented as the
    // size-1 broadcast that leaves the rank alone, beside `insert`.
    (
        "docs/book/src/stdlib.md",
        1,
        "the stdlib entry for the same-rank broadcast, correct after the split",
    ),
    // Two files keep `expand` program text after the deferral rows were
    // removed with the two-candidate model (chelis#1277 S2b). Their operands
    // carry a unit extent at the axis, which is what `expand` means under
    // `spec/04-type-system.md` section 4.7.2, so these are same-rank
    // broadcasts and not migration debt.
    // The Slice B and S2b CLI receipts. Every `expand` here broadcasts a unit
    // axis, or is the negative row that refuses a non-unit one.
    (
        "crates/chelis-cli/tests/runtime_extent_slice_b.rs",
        11,
        "the same-rank broadcast fixture, the zero-extent fixture, the static and runtime non-unit refusals, the unit-extent control, the two locally placed claims over a runtime `shrink` extent, and the two-`expand`-over-one-operand pair with its refuted twin",
    ),
    (
        "crates/chelis-cli/tests/runtime_extent_claim_preparation.rs",
        9,
        "same-rank broadcasts for record projection, literal/symbolic/folded sizes, sum validation, and exported/binding/root acceptance with a runtime unit-operand control and refusal; rank-raising claim fixtures use insert",
    ),
    // `expand`'s own runtime behaviour suite. It exists because the previous
    // `expand.ch` was entirely rank-increasing and moved to `insert.ch`
    // (chelis#1277 S2a), leaving the operation with no suite of its own.
    (
        "packages/chelis-std/tests/runtime/expand.ch",
        6,
        "the chelis-std runtime suite for the same-rank broadcast",
    ),
    // chelis#1506 makes a scalar beside a tensor a type error under
    // `[05-OP-36]`, and the diagnostic names the explicit replacement,
    // `expand(to_tensor([v]), 0i32, shape(xs, 0i32))`. Every site below is
    // that replacement: a unit-extent operand widened at its own axis, which
    // is exactly what `expand` means after the split. None is rank-increasing.
    (
        "crates/chelis-types/src/infer/app.rs",
        1,
        "the replacement spelling inside the `[05-OP-36]` rejection's suggestion",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        1,
        "the same replacement spelling inside the `[05-UNS-1]` runtime refusal",
    ),
    (
        "crates/chelis-cli/tests/coral_prerequisites.rs",
        6,
        "the migration half of the inverted Coral gate: the six programs as Coral must now spell them",
    ),
    (
        "crates/chelis-cli/tests/issue_1506_replacement_spelling_on_the_lanes.rs",
        3,
        "the literal-size, symbolic-size, and folded replacement controls broadcast a unit-extent operand at its existing axis",
    ),
    (
        "crates/chelis-types/tests/issue5_cmp_broadcast_both_forms.rs",
        5,
        "one comparison over two unit-extent broadcasts of `to_tensor([0.5f32])`, plus the four sites of the replacement spelling the rejection names",
    ),
    (
        "crates/chelis-types/tests/issue_942_inferred_tensor_cast.rs",
        14,
        "unit-extent operands at a legal axis, plus one axis-out-of-range negative row",
    ),
    // chelis#668 PR A. The `expand` sites here are POSITIVE CONTROLS: the
    // reproducer PP5's retired validator used to reject is rank 1 beside rank
    // 1 under the one-shape rule, so it is well typed and must stay spelled
    // `expand`. Renaming either to `insert` would delete the control.
    (
        "crates/chelis-types/tests/issue_668_deleted_derivation_does_not_suppress_conv2d.rs",
        3,
        "three `y = expand(x, 0i32, 2i64)` bindings over a unit axis 0, feeding a `conv2d` whose validator must keep running; the operand carries the unit extent, so each is a same-rank broadcast",
    ),
    (
        "crates/chelis-types/tests/issue_668_rank_agreement_is_unification.rs",
        4,
        "one program site, `e = expand(x, 0i32, 2i64)` over a symbolic operand, plus three occurrences of the spelling inside test labels and one assertion message",
    ),
    (
        "crates/chelis-cli/tests/issue_668_elementwise_rank_honesty.rs",
        1,
        "one program site, `e = expand(x, 0i32, 3i64)`, whose refuted unit-extent claim is executed on the evaluator and C lanes to assert §2.4.1's `Domain` trap",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

/// Count `expand` occurrences that are PROGRAM text.
///
/// In a `.rs` file that means inside a string literal, so a Rust identifier
/// such as `fallback_expand_type` or a doc comment is not a call site. The
/// scan is a character state machine over line comments, block comments, raw
/// strings, char literals and normal strings, because a regex for string
/// literals cannot see a `'"'` char literal and one stray quote then swallows
/// the rest of the file.
fn program_sites(text: &str, suffix: &str) -> usize {
    // A Python fixture generator emits Chelis program text from its own string
    // literals, so it is a program source too. `scripts/` was the blind spot
    // that let a rank-raising call survive the migration while its in-tree twin
    // was renamed; counting the whole line is coarse but cannot miss one.
    if suffix == "py" {
        return text.matches("expand(").count();
    }
    if suffix == "ch" {
        return text.matches("expand(").count();
    }
    // Markdown was the scan's last blind spot: a rank-increasing call in a
    // fenced example or an inline span is program text a reader can copy, and
    // two such calls survived the migration in the `openspec/` captures
    // because nothing looked at `.md` at all. Counting the whole call spelling
    // is coarse in the same way the Python rule is -- it also catches a Rust
    // function name such as `emit_expand()` written in prose -- and coarse in
    // the same direction, which is the one that cannot miss a real site.
    if suffix == "md" {
        return text.matches("expand(").count();
    }
    if suffix == "dp" {
        return text.matches("(var {} expand)").count();
    }
    let bytes: Vec<char> = text.chars().collect();
    let mut in_string = false;
    let mut count = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if !in_string {
            if ch == '/' && bytes.get(i + 1) == Some(&'/') {
                while i < bytes.len() && bytes[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if ch == '/' && bytes.get(i + 1) == Some(&'*') {
                let mut depth = 1usize;
                i += 2;
                while i < bytes.len() && depth > 0 {
                    if bytes[i] == '/' && bytes.get(i + 1) == Some(&'*') {
                        depth += 1;
                        i += 2;
                    } else if bytes[i] == '*' && bytes.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                continue;
            }
            if ch == '\'' {
                // A char literal or a lifetime; neither can open a string.
                i += if bytes.get(i + 1) == Some(&'\\') {
                    4
                } else {
                    1
                };
                continue;
            }
            if ch == '"' {
                in_string = true;
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }
        if ch == '\\' {
            i += 2;
            continue;
        }
        if ch == '"' {
            in_string = false;
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(&['e', 'x', 'p', 'a', 'n', 'd', '(']) {
            count += 1;
            i += 7;
            continue;
        }
        i += 1;
    }
    count
}

#[test]
fn every_surviving_expand_program_site_is_inventoried() {
    let root = repo_root();
    let listed = Command::new("git")
        .arg("ls-files")
        .current_dir(&root)
        .output()
        .expect("git ls-files runs in the repository");
    assert!(listed.status.success(), "git ls-files failed");
    let listed = String::from_utf8(listed.stdout).expect("git ls-files emits utf-8");

    let mut measured: Vec<(String, usize)> = Vec::new();
    let mut hull_sites = 0usize;
    for rel in listed.lines() {
        // `spec/` is the language definition, not a program corpus, and
        // chelis#1532 owns what it says.
        if rel.starts_with("spec/") {
            continue;
        }
        // This file holds the needle it searches for, in the literals that do
        // the searching. Counting itself would pin its own implementation.
        if rel == "crates/chelis-types/tests/expand_call_site_inventory.rs" {
            continue;
        }
        let suffix = match rel.rsplit_once('.') {
            Some((_, s)) if matches!(s, "ch" | "dp" | "rs" | "py" | "md") => s,
            _ => continue,
        };
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        if !text.contains("expand") {
            continue;
        }
        let count = program_sites(&text, suffix);
        if count == 0 {
            continue;
        }
        // The Hull corpus is one directory row, not 35 file rows: its
        // disposition is a single upstream fact, and a per-file list would
        // churn whenever upstream regenerates the corpus.
        if rel.starts_with("tests/conformance/hull/programs/") {
            hull_sites += count;
            continue;
        }
        measured.push((rel.to_string(), count));
    }
    if hull_sites > 0 {
        measured.push(("tests/conformance/hull/programs".to_string(), hull_sites));
    }
    measured.sort();

    let mut expected: Vec<(String, usize)> = ALLOWED
        .iter()
        .map(|(path, count, _)| ((*path).to_string(), *count))
        .collect();
    expected.sort();

    assert_eq!(
        measured, expected,
        "the surviving `expand` program sites moved. A new site outside this \
         inventory is a rank-increasing call that should spell `insert`, or a \
         same-rank broadcast that belongs in the inventory with its reason. A \
         missing site means a row here is stale."
    );
}
