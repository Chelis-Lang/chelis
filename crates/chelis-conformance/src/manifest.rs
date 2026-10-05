//! The conformance manifest: `spec/design/shell_repo_contract.md` §11 as
//! machine-readable data.
//!
//! The §11 table is the normative bootstrap checklist and self-audit for every
//! downstream shell repo. Encoding it as [`MANIFEST`] is what lets the audit
//! engine iterate rows deterministically instead of re-deriving them from prose.
//!
//! The doc is the source of truth for the *content*; this table is the machine
//! form. `tests/manifest_tripwire.rs` parses the §11 table and asserts every
//! row's number, contract section, and tier match this data — so a contract
//! change that renumbers a row, moves its section, or changes its tier fails the
//! build until this table is updated in lockstep. This is the
//! `compiler_pin_tripwire.rs` pattern applied to the contract itself.

/// The requirement strength of a contract row. Conditional tiers state their
/// trigger; once triggered they are MUST (contract front matter, RFC-2119).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Unconditional MUST.
    Must,
    /// SHOULD (recommended, not gating).
    Should,
    /// MUST once the shell has at least one expressible upstream blocker (row 11).
    MustIfBlocker,
    /// MUST if the shell validates against external oracles (row 14).
    MustIfExternalOracles,
    /// MUST if the shell links chelis crates as Cargo path deps (row 17).
    MustIfCrateLinking,
}

impl Tier {
    /// Stable machine tag for the tier.
    pub fn tag(&self) -> &'static str {
        match self {
            Tier::Must => "must",
            Tier::Should => "should",
            Tier::MustIfBlocker => "must-if-blocker",
            Tier::MustIfExternalOracles => "must-if-external-oracles",
            Tier::MustIfCrateLinking => "must-if-crate-linking",
        }
    }

    /// Whether a `Fail` on this tier gates the audit (everything but `Should`;
    /// conditional tiers gate once their trigger fires, and the row emits `Na`
    /// when it does not, so treating them as gating here is correct).
    pub fn gates(&self) -> bool {
        !matches!(self, Tier::Should)
    }
}

/// One row of the §11 conformance table.
#[derive(Debug, Clone, Copy)]
pub struct ContractRow {
    /// 1-based row number, matching the `#` column of the §11 table.
    pub row: u8,
    /// Stable internal dispatch key (kebab-case). Not in the doc; used by the
    /// audit engine to route each row to its check.
    pub key: &'static str,
    /// Human artifact description (paraphrase of the table's Artifact column).
    pub artifact: &'static str,
    /// Requirement strength.
    pub tier: Tier,
    /// Owning contract section, e.g. `"§1"`. Matches the table's `Contract §`
    /// column.
    pub section: &'static str,
    /// The chelis release in which this row became a shell requirement. Used by
    /// the audit to downgrade not-yet-applicable rows to informational when
    /// auditing a shell pinned below `since_version` (e.g. the HEAD-binary
    /// canary auditing a stale shell). Rust-only until the contract grows a
    /// matching column; see the contract-amendment change set.
    pub since_version: &'static str,
}

/// The chelis release the shell contract was first made binding in. Every
/// original row carries this as its `since_version` baseline.
pub const CONTRACT_BASELINE_VERSION: &str = "0.7.0";

/// `spec/design/shell_repo_contract.md` §11, row for row. Order matches the
/// table. Keep in lockstep with the doc — `manifest_tripwire.rs` enforces it.
pub const MANIFEST: &[ContractRow] = &[
    ContractRow {
        row: 1,
        key: "agents-md",
        artifact: "AGENTS.md (+ CLAUDE.md symlink) with required sections + intent",
        tier: Tier::Must,
        section: "§1",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 2,
        key: "reef-pin",
        artifact: "reef.toml exact pin = latest validation-clean release",
        tier: Tier::Must,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 3,
        key: "workflow-env-pins",
        artifact: "Workflow env pins in every toolchain-installing workflow",
        tier: Tier::Must,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 4,
        key: "pin-consistency-guard",
        artifact: "Offline pin-consistency CI guard",
        tier: Tier::Must,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 5,
        key: "toolchain-installer",
        artifact: "Toolchain installer + pin-resolving launcher (no global-default side effects)",
        tier: Tier::Must,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 6,
        key: "uv-python",
        artifact: "uv-only Python (stdlib scripts; uv projects for dep-bearing harnesses)",
        tier: Tier::Must,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 7,
        key: "chelis-surface",
        artifact: "docs/CHELIS_SURFACE.md carrying the pinned surface guide in a managed block",
        tier: Tier::Must,
        section: "§3",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 8,
        key: "upstream-bugs",
        artifact: "docs/UPSTREAM_BUGS.md (sections + cadence)",
        tier: Tier::Must,
        section: "§4",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 9,
        key: "staleness-audit",
        artifact: "Citation staleness audit script",
        tier: Tier::Must,
        section: "§4",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 10,
        key: "tests-neg",
        artifact: "tests_neg/ + runner, in CI",
        tier: Tier::Must,
        section: "§6",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 11,
        key: "tests-blocked",
        artifact: "tests_blocked/ + runner, in CI",
        tier: Tier::MustIfBlocker,
        section: "§5",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 12,
        key: "pin-bump-checklist",
        artifact: "Pin Bump Checklist in AGENTS.md",
        tier: Tier::Must,
        section: "§7",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 13,
        // Narrowed to what the audit actually enforces (chelis#739): §8 also
        // requires *mirrored* `.claude/commands` + `.codex/commands` wrappers with
        // the `red-team` alias wired to `redteam-exec`, but neither `conform init`
        // nor `sync` materializes command dirs today, so a fresh scaffold has none
        // and there is nothing to byte-check. Making commands a checked artifact
        // requires embedding a canonical command set (parallel to the skills
        // machinery) so `init`/`sync` create them and the audit can enforce the
        // mirror — deferred as a follow-up. The §8 / §11 contract text keeps the
        // full requirement; this string reflects current enforcement. The
        // `manifest_matches_contract_doc` tripwire compares only row/tier/section,
        // so this description narrowing does not drift the machine form.
        key: "vendored-skills",
        artifact: "Uniform vendored shared skill set + agent skill symlinks",
        tier: Tier::Must,
        section: "§8",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 14,
        key: "parity-harness",
        artifact: "Parity harness (own uv project, checked-in goldens, oracle guards)",
        tier: Tier::MustIfExternalOracles,
        section: "§9",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 15,
        key: "two-config-acceptance",
        artifact: "≥2-config acceptance for new public surface",
        tier: Tier::Must,
        section: "§9",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 16,
        key: "scaffolding-drift-rule",
        artifact: "Scaffolding Drift Rule in AGENTS.md",
        tier: Tier::Must,
        section: "§10",
        since_version: CONTRACT_BASELINE_VERSION,
    },
    ContractRow {
        row: 17,
        key: "chelis-src",
        artifact: "[chelis-src] + chelis reef src store/symlink + local drift guard",
        tier: Tier::MustIfCrateLinking,
        section: "§2",
        since_version: CONTRACT_BASELINE_VERSION,
    },
];

/// Look up a row by its stable dispatch key.
pub fn row(key: &str) -> Option<&'static ContractRow> {
    MANIFEST.iter().find(|r| r.key == key)
}
