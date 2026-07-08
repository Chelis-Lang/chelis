//! Tripwires locking the machine-readable data to its prose/YAML sources.
//!
//! `MANIFEST` (contract §11) and `REGISTRY` (the shell list) are the data forms
//! of two documents that must not drift from them:
//!   - `spec/design/shell_repo_contract.md` §11 conformance table, and
//!   - `.github/workflows/ecosystem-drift.yml`'s canary matrix.
//!
//! These tests parse those sources and assert the Rust data matches, so a
//! contract change (renumber a row, move a section, change a tier) or a canary
//! change (add/remove a shell leg) fails the build until the data is updated in
//! lockstep — the `compiler_pin_tripwire.rs` pattern applied to the contract.

use std::path::{Path, PathBuf};

use chelis_conformance::manifest::{MANIFEST, Tier};
use chelis_conformance::registry;

/// Repo root resolved from `crates/chelis-conformance`'s `CARGO_MANIFEST_DIR`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root must exist")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// Normalize a §11 table Tier cell to the [`Tier`] it denotes. Conditional
/// tiers are matched by keyword so wording tweaks in the doc do not break the
/// mapping, but a genuine tier *class* change (e.g. MUST → SHOULD) does.
fn parse_tier(cell: &str) -> Tier {
    let t = cell.replace('*', "").to_ascii_lowercase();
    if t.contains("should") {
        Tier::Should
    } else if t.contains("blocker") {
        Tier::MustIfBlocker
    } else if t.contains("external oracle") {
        Tier::MustIfExternalOracles
    } else if t.contains("links chelis crates") || t.contains("cargo path dep") {
        Tier::MustIfCrateLinking
    } else {
        assert!(
            t.contains("must"),
            "unrecognized tier cell {cell:?} in shell_repo_contract.md §11"
        );
        Tier::Must
    }
}

/// A parsed §11 data row: (row number, tier cell, contract-section cell).
fn parse_section_11_rows(doc: &str) -> Vec<(u8, Tier, String)> {
    let mut lines = doc.lines();
    // Advance to the §11 table header.
    let header_found = lines.by_ref().any(|l| {
        let t = l.trim_start();
        t.starts_with("| #")
            && t.contains("Artifact")
            && t.contains("Tier")
            && t.contains("Contract")
    });
    assert!(
        header_found,
        "could not find the §11 conformance table header in shell_repo_contract.md"
    );
    // Skip the `|---|---|...` separator row.
    let sep = lines.next().unwrap_or("");
    assert!(
        sep.trim_start().starts_with("|-") || sep.trim_start().starts_with("| -"),
        "expected the §11 header to be followed by a table separator, got {sep:?}"
    );

    let mut rows = Vec::new();
    for line in lines {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('|') {
            break; // end of the table
        }
        let cells: Vec<String> = trimmed.split('|').map(|c| c.trim().to_string()).collect();
        // cells[0] is the empty pre-first-pipe cell; the real columns start at 1.
        // Layout: | # | Artifact | Tier | Contract § | School exemplar |
        assert!(
            cells.len() >= 6,
            "malformed §11 row (need >=5 columns): {line:?}"
        );
        let row_num: u8 = cells[1]
            .parse()
            .unwrap_or_else(|_| panic!("§11 row number is not an integer: {:?}", cells[1]));
        let tier = parse_tier(&cells[3]);
        let section = cells[4].clone();
        rows.push((row_num, tier, section));
    }
    rows
}

#[test]
fn manifest_matches_contract_doc() {
    let doc = read(&repo_root().join("spec/design/shell_repo_contract.md"));
    let parsed = parse_section_11_rows(&doc);

    assert_eq!(
        parsed.len(),
        MANIFEST.len(),
        "shell_repo_contract.md §11 has {} rows but MANIFEST has {}. \
         Update crates/chelis-conformance/src/manifest.rs to match the table.",
        parsed.len(),
        MANIFEST.len()
    );

    let mut mismatches = Vec::new();
    for (i, ((doc_row, doc_tier, doc_section), row)) in parsed.iter().zip(MANIFEST).enumerate() {
        if *doc_row != row.row {
            mismatches.push(format!(
                "index {i}: doc row # {doc_row} vs MANIFEST row {}",
                row.row
            ));
        }
        if doc_section != row.section {
            mismatches.push(format!(
                "row {doc_row}: doc section {doc_section:?} vs MANIFEST {:?}",
                row.section
            ));
        }
        if *doc_tier != row.tier {
            mismatches.push(format!(
                "row {doc_row}: doc tier {doc_tier:?} vs MANIFEST {:?}",
                row.tier
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "MANIFEST drifted from shell_repo_contract.md §11:\n  {}\n\n\
         Fix crates/chelis-conformance/src/manifest.rs to match the §11 table \
         (or update the table if the contract intentionally changed).",
        mismatches.join("\n  ")
    );
}

#[test]
fn registry_matches_drift_matrix() {
    let yaml = read(&repo_root().join(".github/workflows/ecosystem-drift.yml"));

    // Every matrix leg is a `- repo: <name>` line; nothing else in the file uses
    // that exact key (checkouts use `repository:`, tokens use `repositories:`).
    let mut matrix: Vec<String> = yaml
        .lines()
        .filter_map(|l| {
            let t = l.trim_start();
            t.strip_prefix("- repo:")
                .map(|rest| rest.trim().to_string())
        })
        .collect();
    matrix.sort();
    matrix.dedup();

    assert!(
        !matrix.is_empty(),
        "found no `- repo:` matrix legs in ecosystem-drift.yml; the parser or the \
         workflow layout changed"
    );

    let mut active: Vec<String> = registry::active_shells()
        .map(|s| s.name.to_string())
        .collect();
    active.sort();

    let only_in_matrix: Vec<&String> = matrix.iter().filter(|m| !active.contains(m)).collect();
    let only_in_registry: Vec<&String> = active.iter().filter(|a| !matrix.contains(a)).collect();

    assert!(
        only_in_matrix.is_empty() && only_in_registry.is_empty(),
        "REGISTRY active shells and the ecosystem-drift.yml matrix disagree:\n  \
         only in canary matrix: {only_in_matrix:?}\n  \
         only in REGISTRY (status Active): {only_in_registry:?}\n\n\
         Reconcile crates/chelis-conformance/src/registry.rs with the canary \
         matrix (the ground-truth active set).",
    );
}
