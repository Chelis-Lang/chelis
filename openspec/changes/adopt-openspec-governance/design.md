## Context

`spec/design/spec_provenance.md` defines Phase 0 as OpenSpec configuration, agent instructions, strict validation, human review routing, citation checks, a temporary exemption path, and a PR gate. It explicitly leaves the phase inactive until its adoption and PR-gate oracles are green. Chelis already has strong specification hierarchy, test-first and negative-test rules, one-oracle completion discipline, and Python automation conventions, but it previously had no canonical `openspec/` planning root.

The private `Chelis-Lang/ci` repository now exposes an input-free `actions/openspec-governance` composite action. Commit `2906e03880a2ea7d553b0959c246991b1b058990` pins Node 24.18.0, an integrity-locked OpenSpec 1.6.0 npm graph, bounded GitHub event-base selection, and a fixed call to the checked-out consumer's `scripts/check_openspec.py`. The action intentionally owns no Chelis policy, checkout, secret, permission, artifact, or acceptance decision.

Chelis's existing `CI` workflow grants workflow-wide `contents: write`, while OpenSpec needs only read access. The governance path therefore needs an explicit least-privilege boundary rather than inheriting unrelated release or maintenance authority.

## Goals / Non-Goals

**Goals:**

- Materialize the Phase 0 planning policy as testable OpenSpec requirements without copying or weakening `spec/**`.
- Require a visible planning-only ancestor before governed implementation and preserve one lifecycle per branch.
- Provide a narrow repository-owned maintenance classification that can never bypass `spec/**`.
- Implement one stdlib Python checker used by local validation and the immutable shared GitHub Action.
- Prove every control with positive and planted negative fixtures, then require exact Chelis-hosted evidence before activation.

**Non-Goals:**

- Change the Chelis language, compiler, runtime, CLI, backends, packages, numerical semantics, generated artifacts, or existing acceptance oracles.
- Make OpenSpec or GitHub state canonical product, atom, freshness, coverage, provenance, or implementation-correctness authority.
- Implement Buoy, a Chelis Buoy adapter, atom revisioning, coverage policy, or impact closure.
- Copy the shared action's Node/npm implementation into Chelis or add Nix to required GitHub Actions.
- Retroactively rewrite historical branches or atomize existing specifications.

## Decisions

### 1. Use the built-in spec-driven schema under the canonical root

Chelis uses `openspec/config.yaml`, `openspec/specs/`, and `openspec/changes/` with the built-in `spec-driven` schema. Project-local schema overrides are rejected because they could weaken proposal/spec/design/task completion. The old empty `.openspec/` directory is not an authority and is outside the current CLI contract.

The configuration summarizes Chelis-specific boundaries and artifact rules. It points agents back to `AGENTS.md`, the specification hierarchy, and `spec/design/spec_provenance.md` rather than reproducing those authorities.

**Alternative rejected:** A custom Phase 0 schema. It adds an override surface before the standard lifecycle has any adoption evidence and makes the shared checker harder to reason about.

### 2. Make planning order a Git property

A governed branch adds one lifecycle. A planning-only commit containing the lifecycle marker, proposal, and requirement deltas must be an ancestor of the first production change. This gives the checker repository-owned, provider-independent ordering evidence and makes a collapsed "plan plus implementation" commit fail visibly.

For this adoption branch the planning-only ancestor is commit `1297dfac`, and the pull-request citation format is one body line starting at column one reading exactly `OpenSpec-Change: <change-id>`; the ordering and routing fixtures in `scripts/test_check_openspec.py` encode both.

The pull-request body still cites the exact change identifier for human routing. That citation and GitHub review state are process evidence; the Git history and committed artifacts remain the durable validation inputs. The checker reads only the bounded event payload already available to the job and uses no API token.

**Alternative rejected:** Trusting commit-message trailers, labels, timestamps, or live review queries as the sole proof that planning preceded implementation. They are mutable, spoofable, unavailable on some events, or require broader credentials.

### 3. Classify every normative spec edit as governed

Every path under `spec/**` is normative for Phase 0 routing even when it is Markdown. Nonnormative documentation and behavior-preserving maintenance outside that tree may use one exact-path TOML manifest under `openspec/exemptions/`. The checker compares its declared paths with the full branch diff and rejects symlinks, invalid dates, governance paths, `spec/**`, mixed lifecycle/exemption evidence, and scope expansion.

**Alternative rejected:** A `spec-exempt` label or generic docs-only detector. Provider state is not a durable exemption ledger, and a broad Markdown bypass contradicts the existing Phase 0 contract.

### 4. Require synchronized archive state at merge

Local pre-archive validation may accept one complete active lifecycle while requiring unchanged baseline `openspec/specs/`. Merge-bound validation accepts no active lifecycle: it requires a date-stamped archive, complete described tasks, strict validation, and baseline specs that equal replaying the archived delta against the comparison-base specs.

This keeps proposal-time deltas reviewable and prevents early baseline edits from masking archive collisions. OpenSpec completion does not replace the owning Chelis executable oracle named by the tasks.

**Alternative rejected:** Allowing active changes to merge or trusting an archive move without replaying synchronization. Either permits unfinished intent or stale baseline requirements to report green.

### 5. Keep one Chelis-owned Python checker

`scripts/check_openspec.py` is stdlib-only Python 3.11+ and exposes pure functions suitable for `scripts/test_check_openspec.py`. It has explicit local pre-archive and merge-bound modes, accepts the central action's combined `--self-test --merge-bound --base` invocation, honors injected `GIT_BIN` and `OPENSPEC_BIN`, and independently verifies exact OpenSpec 1.6.0.

Self-test fixtures mutate temporary copies and must demonstrate failure for each critical control: malformed specs, absent negative scenarios, artifact drift, planning-order collapse, citation mismatch, branch-scope ambiguity, invalid exemptions, symlinks, unchecked tasks, active merge state, malformed archives, and unsynchronized deltas.

**Alternative rejected:** Copying the central hosted fixture checker. That file proves launcher mechanics only and contains no Chelis policy.

### 6. Isolate the immutable GitHub integration

A dedicated `openspec-governance.yml` workflow uses `contents: read`, full-history checkout via `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1`, disabled credential persistence, and `Chelis-Lang/ci/actions/openspec-governance@2906e03880a2ea7d553b0959c246991b1b058990`. It passes no inputs or secrets and does not duplicate Node, npm, event, or launcher mechanics.

The pinned action SHA is the independently recorded accepted central revision: on 2026-07-24 its deterministic contract suite (18 tests) passed at a `2906e038` checkout, the hosted `openspec-governance-contract` lane was green at the same commit (<https://github.com/Chelis-Lang/ci/actions/runs/30116322777>), and `Chelis-Lang/ci` Actions access was verified organization-wide. Any later pin change requires re-recording this evidence.

A separate workflow is preferred over inheriting `ci.yml`'s current workflow-wide write permission. Branch protection is updated only after observing the exact hosted context from the final archived adoption branch.

**Alternative rejected:** `@main`, a mutable release tag, copied npm provisioning, or a required Nix bootstrap. Each weakens provenance or adds an unrelated failure domain.

### 7. Activation occurs at an evidence-gated merge

This branch may prepare the complete surface while policy remains pending. Before merge, the lifecycle is synchronized and archived, the deterministic `openspec_adoption` and checker suites are green, and the PR runs the exact private action plus the complete Chelis hosted suite. The merge itself is the activation point; no earlier artifact creation claims enforcement.

## Risks / Trade-offs

- **[Draft implementation PRs remain red in merge-bound mode while their lifecycle is active]** → Document and provide a local pre-archive mode; require green only after final synchronization and archive.
- **[The exact maintenance path creates ceremony for tiny edits]** → Keep the TOML schema minimal and deterministic; never relax the mandatory `spec/**` lifecycle path.
- **[A large checker becomes a second policy language]** → Derive every control from these specs, keep functions dependency-free and testable, and reject unrelated provenance or compiler semantics.
- **[Private action access fails only on GitHub]** → Treat the exact consumer-hosted run as required evidence and retain the unactivated workflow state for rollback.
- **[Current branches predate activation]** → Apply the policy only after the evidence-gated merge and document how already-open branches acquire one lifecycle or exemption before their next merge-bound run.

## Migration Plan

1. Initialize the canonical OpenSpec root and create this adoption lifecycle while Phase 0 remains inactive.
2. Review and accept the proposal and requirement deltas before any checker or workflow implementation.
3. Add planted negative fixtures and `scripts/test_check_openspec.py`, then implement the minimum checker functions that satisfy them.
4. Add contributor or agent guidance, PR citation guidance, `spec/**` review routing, and the repository-owned maintenance path.
5. Add the isolated read-only workflow pinned to the reviewed checkout and central action SHAs.
6. Run strict OpenSpec validation, the named adoption/checker suites, and the existing relevant script tests; fix every falsification before archival.
7. Synchronize and archive this lifecycle, run the exact hosted PR and complete Chelis suite, and merge only while all required evidence is green.
8. After observing the stable hosted check name, update branch protection deliberately. Roll back by restoring the pre-adoption workflow or a separately reviewed known-good action SHA; do not change compiler behavior or specification authority during rollback.

## Open Questions

None. The implementation may refine diagnostics and fixture organization without changing the fixed ownership, ordering, classification, archive, or activation contracts above.
