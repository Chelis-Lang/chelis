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
//! The two groups below have different futures. The deferral files go away with
//! the mechanism, at which point their rows come out and the remaining
//! inventory is the same-rank sites alone. The same-rank sites stay, and each
//! one is a program whose declared result has the operand's own rank over a
//! unit extent at the axis.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Files that may still contain `expand` program text, with the exact count and
/// the reason the sites are there.
const ALLOWED: &[(&str, usize, &str)] = &[
    // Genuine same-rank broadcasts: the declared result has the operand's rank.
    // These are what `expand` means after the split, so they are not renamed.
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
    // Deferral files: their subject is the choice between the two candidate
    // forms, which `insert` does not have. The removal slices delete them, and
    // these rows come out with them.
    (
        "crates/chelis-cli/tests/hash_order_stability.rs",
        13,
        "deferral: hash-order settlement of coupled candidate choices",
    ),
    (
        "crates/chelis-types/tests/issue5_cmp_broadcast_both_forms.rs",
        10,
        "deferral: the comparison family selecting a candidate",
    ),
    (
        "crates/chelis-types/tests/issue_1380_matmul_candidate_elimination.rs",
        12,
        "deferral: candidate elimination",
    ),
    (
        "crates/chelis-types/tests/issue_942_inferred_tensor_cast.rs",
        44,
        "deferral: which candidate a later consumer selects",
    ),
    (
        "crates/chelis-types/tests/slice_c_builtin_settlement_tripwire.rs",
        4,
        "deferral: the per-builtin settlement registry",
    ),
    (
        "crates/chelis-types/tests/slice_c_composite_carriers.rs",
        19,
        "deferral: carriers of an unmade choice",
    ),
    (
        "crates/chelis-types/tests/slice_c_constrain_contexts.rs",
        18,
        "deferral: contexts that constrain a pending result",
    ),
    (
        "crates/chelis-types/tests/slice_c_freeze_rows.rs",
        13,
        "deferral: the freeze default between two candidates",
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
    if suffix == "ch" {
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
    for rel in listed.lines() {
        // `spec/` is the language definition, not a program corpus, and
        // chelis#1532 owns what it says.
        if rel.starts_with("spec/") {
            continue;
        }
        let suffix = match rel.rsplit_once('.') {
            Some((_, s)) if matches!(s, "ch" | "dp" | "rs") => s,
            _ => continue,
        };
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        if !text.contains("expand") {
            continue;
        }
        let count = program_sites(&text, suffix);
        if count > 0 {
            measured.push((rel.to_string(), count));
        }
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
