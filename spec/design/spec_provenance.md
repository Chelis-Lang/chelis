# Spec Provenance: atoms, hash-linked claims, and the coverage gate

**Status:** Design proposal, pre-implementation. Tracking issue: chelis#733.
**Owning specs:** every file under `spec/` (this plan changes how normative
text in them is WRITTEN and REFERENCED, not what any of them says);
`spec/design/ears_chelis_bridge.md` (the existing EARS anchor, developed in
concert - see open question 1); the repo Contract Invariants and the
Spec-First Development section of `CLAUDE.md` (this plan is that section
made machine-checkable).
**Class fixed:** the recurring failure BENEATH the four numeric-audit
classes: **spec silence and stale spec claims**. `spec/04-type-system.md`
said nothing about integer overflow, which is why int8-wraps and
int64-saturates coexisted unauthored (chelis#680/#718); integer `mean`
(#724) and bool arithmetic (#726) were never decided by anyone; and four
code comments asserted safety properties the code lacked (#694) because
nothing invalidates a prose claim when reality moves.
**Sibling plans:** chelis#729/#730/#731/#732 fix the four classes; this
plan is the fifth and sits underneath them - it makes "there is no spec
for this" and "the spec moved under this claim" mechanically detectable,
which none of the four can do for themselves.

## Summary

Chelis already runs a doc-to-code tripwire in production: `chelis reef
conform` locks a machine-readable MANIFEST to the shell-contract document
and fails CI on drift. This plan points the same idea inward at `spec/`:

1. **Atoms**: normative statements in spec files become addressable blocks
   with stable IDs and content hashes (EARS-shaped where it fits).
2. **Hash-linked claims**: tests (primarily) and code (secondarily) carry
   `@spec <ID> rev <hash>` annotations; `chelis-lint` - which already
   parses Rust source repo-wide (the §8.6 rule) and runs in the gate -
   verifies that every referenced atom exists and that its hash still
   matches the atom's current text. **A spec edit automatically invalidates
   every claim made against the old text**, and the gate stays red until
   each carrier is re-affirmed or honestly downgraded.
3. **The coverage gate**: atoms must be carried by tests (spec-first made
   checkable); PRs must cite the atoms or design doc they build against,
   or carry an explicitly signed-off exemption; and structural surface
   (capability-table rows, tag dispositions, tolerance rows) must cite
   atoms once the sibling plans' tables exist.

**The asymmetric-hashing decision** (the load-carrying design choice):
hash the SPEC side only; let TESTS pin the code side. Code churns
constantly - hashing code bodies (the full Unison move) produces either
noise or nothing. Spec text is stable, and behavior is already pinned by
the artifact this repo produces in abundance: executable tests. So the
enforceable triple is **atom <-> test <-> code**, where the atom-test edge
is machine-checked (existence, freshness, coverage) and the test-code edge
is what tests already are.

**Scope verdict:** no new external tooling (the openspec.dev / EARS ideas
are adopted as *formats and flows*, implemented with the repo's own
machinery: chelis-lint, the gate, CI, Python scripts with tests per repo
policy). One new lint rule family, one atom grammar, one CI job, and an
incremental atomization that starts where the four sibling plans are about
to write normative text anyway - their spec deliverables are BORN
atomized, so the system grows with the work instead of demanding a
big-bang retrofit.

## What this plan can and cannot enforce (read first)

It enforces that **claims exist, are specific, and stay fresh**. It cannot
enforce that claims are **true** - an agent can paste an `@spec`
annotation onto code that does not conform. Three things bound that
honestly:

1. the required carrier is a **test** (a test that executes the
   requirement is hard to annotate falsely and stay green);
2. the sibling plans' oracles (matrix tests, invariants, generated
   conformance) are where truth lives - this plan routes attention, it
   does not replace execution ("execute everything" remains the law);
3. reviewer effort is re-aimed, not eliminated: checking "does this test
   actually exercise atom 04-OVF-3" is a far smaller ask than "is this
   PR's spec story coherent at all", which is what reviews silently carry
   today.

## Non-goals

- **Not** a retrofit of all thirteen spec files at once. Atomization is
  incremental and demand-driven (§C1 scope rules); un-atomized prose
  stays valid editorial text.
- **Not** code content-addressing (Unison's language-level move). Spec-
  side hashes only, per the asymmetric-hashing decision.
- **Not** an external requirements tool or format lock-in. EARS shapes
  the atom grammar; openspec.dev's spec-delta-before-code flow shapes the
  PR gate; both are implemented in-repo.
- **Not** a replacement for the four sibling plans' enforcement. The
  capability table still makes undecided cells unbuildable (#729); this
  plan makes the table's rows CITE their authority and makes authoring
  that authority a visible, gated act.
- **Not** prose policing. Only text inside atom blocks is normative,
  hashed, and protected; everything else in `spec/` remains freely
  editable documentation.

## Vocabulary

- **Atom** - one addressable normative statement: an ID, a hash, and a
  block of SHALL-text. The unit of citation, coverage, and freshness.
- **Rev** - the truncated content hash of an atom's normalized text,
  recorded at claim sites. Stale rev = the spec moved under the claim.
- **Carrier** - an annotated site. *Test carriers* (in `tests/`,
  `#[test]` functions, probe fixtures) are load-carrying for coverage;
  *code carriers* (implementation sites) are navigation/ownership
  metadata and optional.
- **Coverage edge** - the atom-to-test-carrier requirement, enforced per
  the blocking manifest (§C4).
- **Tombstone** - a retired atom ID kept in a registry with its
  disposition (superseded-by, withdrawn), so stale references produce a
  pointed message instead of "unknown ID". IDs are never reused.
- **The exemption** - the signed-off escape hatch for PRs that genuinely
  precede spec (prototypes, investigations); loud, labeled, and tracked.

---

# Part I - the normative contracts (§C1-§C5)

## C1. The atom grammar

An atom is a Markdown blockquote block, machine-extractable, inside any
`spec/*.md` or `spec/design/*.md` file:

```markdown
> **[04-OVF-3]** WHEN an integer op result exceeds its declared width,
> every lane SHALL trap with the branded overflow diagnostic
> (`numeric trap: overflow in <op> at <prim>`); no lane SHALL wrap or
> saturate.
```

Rules:

1. **ID**: `[<file-prefix>-<topic>-<n>]` - file prefix from the owning
   spec (`04`, `05`, `design/dtype` for design docs...; exact scheme
   fixed in Phase 1), topic slug, monotonically increasing `n`. IDs are
   immutable and never reused; retirement goes through the tombstone
   registry (§C3.4).
2. **Shape**: EARS-style where the statement is behavioral (WHEN/WHILE
   trigger + SHALL response); plain declarative SHALL-text where it is
   structural ("the Deep vocabulary SHALL comprise exactly the 62 tags
   listed in ..."). MUST/SHALL only inside atoms; aspirational prose
   stays outside them.
3. **Hash**: SHA-256 of the whitespace-normalized atom text (collapse
   runs of whitespace, strip the ID marker itself), truncated to 8 hex
   chars. Normalization means reflowing or re-wrapping an atom does NOT
   invalidate claims; changing any word does.
4. **Extent**: the blockquote is the atom. Tables can be atoms (a
   blockquoted table, e.g. the tolerance table's rows); one atom per
   decision, not per paragraph of exposition.
5. **Scope of atomization** (the incremental rule): (a) all NEW normative
   text lands atomized from Phase 1 onward - in particular the spec
   sections the sibling plans deliver (#729's overflow/rounding section,
   #732's tolerance table, #731's disposition rules, #730's diagnostic
   format) are born as atoms; (b) EXISTING text is atomized when it is
   first cited, amended, or found load-bearing by an audit - never as a
   standalone bulk pass. Un-atomized normative prose is a known-debt
   report line (§C4.3), not an error.

## C2. The claim annotation and the lint rules

Carrier syntax, valid in Rust, Python, `.ch`/`.dp` fixtures, and Markdown
(comment syntax of the host language):

```rust
// @spec 04-OVF-3 rev 9f3ac2d1
// @spec 05-TOL-1 rev 04b77e10   (multiple atoms per site allowed)
#[test]
fn int8_add_overflow_traps() { ... }
```

The `spec-provenance` lint family (in `chelis-lint`, blocking in the gate
like §8.6):

1. **`spec-ref-valid`**: every `@spec` ID resolves to a live atom or a
   tombstone. A tombstone hit reports the disposition ("superseded by
   04-OVF-7; re-read and re-point"). An unknown ID is an error.
2. **`spec-ref-fresh`**: the recorded rev equals the atom's current hash.
   A stale rev is an error naming the atom, its file/line, and both
   hashes. There is deliberately NO auto-fix: the fix is a human/agent
   act - re-read the atom, then either re-affirm (update the rev - the
   carrier still conforms) or downgrade honestly (the carrier no longer
   conforms: flip the test red/`#[ignore]` with an issue, per the
   sibling plans' red-to-green discipline).
3. **`spec-atom-wellformed`**: atom blocks parse, IDs are unique
   repo-wide, hashes in the tombstone registry are consistent.
4. **One parser**: the atom extractor lives in `chelis-lint` (Rust),
   which both enforces and exports; a `--spec-report` mode dumps the
   atom index and coverage data as JSON for CI jobs and scripts (exact
   CLI surface: open question 3). No second parser anywhere (the
   conform-framework lesson: duplicated readers drift).

**The freshness protocol** (the point of the whole mechanism): a PR that
edits an atom's text MUST, in the same change set, visit every carrier of
that atom - the lint enumerates them - and re-affirm or downgrade each.
This is the #694 fix generalized: a safety claim can no longer outlive
the text it was made against. The cost (spec edits fan out) is the
feature: editing normative text SHOULD be a deliberate act that confronts
its consequences, and the normalization rule (§C1.3) keeps purely
editorial edits free.

## C3. The PR spec gate

A CI job (Python, tested, per repo scripting policy) on every PR:

1. The PR description names its authority: a `Spec-Atoms:` trailer
   listing atom IDs it implements/affects, and/or a `Spec-Design:`
   trailer naming a `spec/design/*.md` it adds or amends.
2. PRs with neither fail the job UNLESS they are (a) mechanically
   docs-only, (b) labeled `spec-exempt` - a label only maintainers can
   apply (ruleset-gated), carrying a one-line reason; the job posts the
   exemption into a rolling report so exemptions are visible debt, not
   silence.
3. `spec/**` paths are CODEOWNERS-protected: normative changes always
   cross a human (the signoff the audit's undecided cells never got).
4. The job cross-checks cheaply: cited atom IDs must exist; a PR whose
   diff touches files carrying `@spec` annotations inherits those atoms
   into its expected citation set (advisory note, not a failure, in
   Phase 0; tightened later per §C4).

This tier is deliberately process-level and gameable in isolation - its
job is to make "no spec story at all" impossible and to route reviewer
attention; the lint tier (§C2) and the structural tier (§C5) carry the
mechanical weight.

## C4. The coverage contract

1. **Coverage edge**: every atom in a *blocking-manifest* spec file must
   have at least one green (or issue-linked `#[ignore]`d red) TEST
   carrier. The manifest starts empty, gains `spec/04` and `spec/05`
   sections as Phases 1-2 atomize them, and only ever grows (a ratchet,
   like the tripwire baselines).
2. **The two honest states**: a green test carrier = "specified and
   honored"; an `#[ignore]`d carrier naming an issue = "specified, not
   yet honored" (exactly the audit's red-test discipline - the ~70
   ignored tests on PR #696 become carriers for the atoms the sibling
   plans author). There is no representable state for "specified,
   silently unhonored" - that is the point.
3. **The debt reports** (CI artifacts, advisory): atoms with no carrier;
   normative-looking prose outside atoms (SHALL/MUST outside blockquotes)
   in manifest files; live exemption labels; carrier counts per atom.
   Reports precede ratchets: a section enters the blocking manifest only
   when its report is clean.

## C5. The structural tier (rides the sibling plans)

Where the sibling plans create machine-readable surface, atoms become
build-relevant, which is the strongest form of this plan:

1. **Capability-table rows cite atoms** (#729 Phase 4): every
   `Supported | Rejected(reason)` cell carries the atom ID that decided
   it; the table's conformance generator fails on a row with no citation.
   Undecided-cell bugs (#724, #726) become unwritable-without-an-atom -
   the signoff flow the user-facing question asked for, enforced by the
   build.
2. **Tag dispositions cite atoms** (#731 Phase 3): each `DeepTag`
   variant's checker disposition names its spec/03 atom.
3. **Tolerance rows are atoms** (#732 Phase 3): the per-op cross-lane
   bounds land as a blockquoted atom table in spec/05; the #687 oracle
   reads the same rows the lint hashes.
4. **Diagnostic strings cite atoms** (#730): the frozen `unsupported:`
   and trap message constants carry code-carrier annotations to their
   atoms, so a message edit and its spec move together or the gate says
   why not.

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| atom grammar + ID scheme + hash normalization (§C1) | Phase 1 | this doc + a migration script for existing atoms, one change set |
| lint rule semantics (§C2) | Phase 1 | this doc + lint rule spec registration |
| PR gate requirements (§C3) | Phase 0 (mechanics), Phase 2 (tightening) | this doc |
| blocking manifest contents | grows per §C4.3 | additions only; a removal is an incident |
| tombstone registry | append-only from Phase 1 | never rewritten |

## B2. Invariants that hold across every boundary

1. **Atoms are append-mostly.** Editing an atom's meaning retires the ID
   (tombstone, superseded-by) and mints a new one when the change is
   substantive; the rev mechanism handles wording refinement. Which of
   the two applies is a review judgment - the default for anything a
   carrier might no longer satisfy is retire-and-mint.
2. **No claim without a reader.** A rev may only be updated by a change
   set whose description affirms the re-read ("re-affirmed against
   04-OVF-3 rev 9f3ac2d1"). Mechanical rev-bumping across many sites
   without per-site affirmation is the failure mode; reviews reject it.
3. **Tests are the coverage currency.** Code carriers never satisfy the
   coverage edge; a plan or PR that "covers" an atom with only
   implementation annotations has not covered it.
4. **The sibling plans' controls rules are inherited**: red-to-green by
   un-ignoring; discoveries fork; both-lanes-or-neither for behavioral
   carriers.
5. **Exemptions decay.** A `spec-exempt` PR creates a follow-up
   obligation (write the atoms or the design doc); the rolling report
   (§C3.2) keeps them visible until discharged.

## B3. How to pick up a phase

1. Read Part I, your phase, and the previous phase's frozen-at-exit
   list; §C5 items ship jointly with the named sibling-plan phases -
   coordinate via both tracking issues.
2. Run the debt reports first (once they exist); the deltas are your
   work-list.
3. The four sibling design docs are the first citation targets - when in
   doubt about what to atomize first, atomize what they are about to
   need.
4. Gate with `scripts/gate.py --local`; docs-only stages per the repo's
   docs-only rule.

---

# Part III - the phases

## Phase 0 - the PR spec gate (process tier; afternoon-scale, land-first)

**You inherit:** nothing but this document and the repo's CI.

**You deliver:**

1. The PR-gate CI job (§C3.1-2): trailer parsing, docs-only detection,
   the `spec-exempt` label path, the rolling exemption report. Python,
   with tests, per repo policy.
2. `spec/**` CODEOWNERS protection and the maintainers-only label
   ruleset.
3. The PR template gains the `Spec-Atoms:` / `Spec-Design:` trailers and
   one line pointing here.

**Frozen at your exit:** the gate's requirement set (§C3.1-2).

**Explicitly not yours:** atoms, hashes, the lint, any spec edits.

**Oracle:** a planted PR with no citation and no exemption fails the
job; a `spec-exempt` PR passes and appears in the report; a docs-only PR
passes untouched. (The job's own test suite is the named suite.)

## Phase 1 - atoms, the extractor, and the lint (the mechanism tier)

**You inherit:** the gate (your delivery vehicle) and the sibling plans'
imminent spec deliverables (your first atoms).

**You deliver:**

1. The atom grammar finalized (§C1; resolve open question 1 with the
   EARS bridge doc) and the tombstone registry file.
2. The extractor + `spec-ref-valid` / `spec-ref-fresh` /
   `spec-atom-wellformed` in chelis-lint, with the rule-spec
   registration and the lint crate's standard positive/negative rule
   tests; `--spec-report` JSON export.
3. **First atoms**: spec/04 §9-§10 and spec/05 §7-§8 carry the four
   sibling plans' decided contracts as provisional atoms (seeded ahead
   of this plan) , i.e. ([04-NUM-*], [04-TOT-*], [05-UNS-*], [05-OBS-*])
   with status banners. This phase RATIFIES their grammar against the
   finalized §C1, computes their revs, and additionally atomizes the
   audit-proven load-bearing statements (the §5.7 precision-widening
   rule, the §1.1.1 f8e4m3 deferral, the 62-tag vocabulary sentence).
4. **First carriers**: the audit matrix tests annotated against those
   atoms (they were born from these exact claims; the mapping is
   mechanical).

**Frozen at your exit:** §C1 grammar + §C2 semantics.

**Explicitly not yours:** coverage BLOCKING (Phase 2); bulk atomization.

**Oracle:** the lint green on the tree; red on a planted bogus ID AND a
planted stale rev (the rules' negative tests); the extractor's unit
suite; at least one real spec-edit exercised end-to-end (edit an atom,
watch carriers fail, re-affirm, green).

## Phase 2 - the coverage edge and the first ratchet

**You inherit:** live atoms, live lint, annotated audit tests.

**You deliver:**

1. The coverage computation in `--spec-report` (atoms x test-carriers,
   the two honest states) and the CI debt-report artifact (§C4.3).
2. The blocking manifest with its first entries: the atomized sections
   of spec/04 and spec/05, once their reports are clean.
3. The §C3.4 tightening: PRs touching carrier-bearing files must cite
   those atoms (from advisory to failing).

**Frozen at your exit:** the coverage rules and the manifest ratchet.

**Explicitly not yours:** the structural tier.

**Oracle:** a planted uncovered atom in a manifest file fails the gate;
the debt report runs in CI on every PR; the exemption report shows only
discharged or in-flight entries.

## Phase 3 - the structural tier (jointly with #729/#731/#732)

**You inherit:** the mechanism, the coverage gate, and the sibling
plans' tables as they land.

**You deliver:** §C5's four hooks, each shipped inside the corresponding
sibling-plan phase with this plan's lint checking citation presence:
capability rows cite atoms; tag dispositions cite atoms; tolerance rows
ARE atoms; frozen diagnostic constants carry code annotations.

**Frozen at your exit:** the citation requirements per table.

**Explicitly not yours:** the tables themselves.

**Oracle:** per hook, the mutation test - a table row / variant /
tolerance entry without a citation fails the corresponding build or
lint; recorded once in each joint PR.

## Phase 4 - executable atoms (opportunistic, long-term)

**You inherit:** everything above, plus a trustworthy prove lane
(gated on #688's fix for integer properties).

**You deliver:** semantic atoms mapped to in-language `@property`
declarations carrying `@spec` annotations - spec claims checked by
`chelis prove` rather than sampled by tests, starting with the overflow
and rounding atoms (which are property-shaped by construction). Each
mapping retires no test carriers; properties are additional evidence.

**Oracle:** the property suite per atom set, named in the owning spec
section, run per the repo's manual-gate documentation rules.

---

# Part IV - bookkeeping

## I1. Interlocks

- **#729/#730/#731/#732**: their spec deliverables are born atomized
  (Phase 1 here coordinates with whichever of their phases is in
  flight); their tables are the structural tier (Phase 3 here ships
  inside their table-bearing phases). Nothing here blocks them: if this
  plan lags, they land un-atomized normative text and §C1.5's
  "atomize-on-first-citation" rule catches it later.
- **`chelis reef conform`**: the shell-contract MANIFEST mechanism is
  the architectural precedent; this plan deliberately mirrors its
  tripwire discipline (doc and machine-readable form move in lockstep or
  CI fails). Long-term, `conform`'s own contract document is a
  candidate for the same atom grammar (open question 4).
- **`ears_chelis_bridge.md`**: the atom grammar's EARS shape is decided
  WITH that document, not beside it (open question 1).
- **The agent-QC layer** (nested CLAUDE.md files, change-shape skills,
  mechanism index - per the standing quality discussion): skills like
  `add-a-builtin` gain a step "cite or author the atom"; the PR gate is
  the enforcement behind that step.

## Issue map

| phase | what becomes impossible |
|---|---|
| 0 | a PR with no spec story at all; unsignposted spec edits |
| 1 | stale spec claims (#694's class); citations to nothing |
| 2 | "specified, silently unhonored" atoms in ratcheted files; new spec-silent surface in them |
| 3 | undecided table cells without an authoring act (#724/#726's class, at the build) |
| 4 | (additive) semantic atoms drifting from checked behavior |

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | atom grammar's EARS profile (which EARS templates; how far to push structure into the blockquote) | Phase 1, with `ears_chelis_bridge.md` | §C1 + that doc |
| 2 | ID scheme details (file-prefix stability across spec renumbering; design-doc prefixes) | Phase 1 | §C1.1 |
| 3 | CLI surface for the extractor/report (`chelis lint --spec-report` vs a `chelis spec` subcommand) | Phase 1 | §C2.4 + CLI docs |
| 4 | whether `shell_repo_contract.md` + conform MANIFEST migrate to the atom grammar | after Phase 2, separate proposal | conform docs |
| 5 | rev-affirmation ergonomics for large fan-outs (a spec edit touching 50 carriers): per-site vs per-atom affirmation blocks | Phase 1, revisit Phase 2 | §B2.2 |

## The one-sentence summary for a reviewer

Give every normative sentence an address and a fingerprint, make tests
the citizens that carry them, let the gate refuse claims to nothing and
claims gone stale, and require every PR to name its authority or visibly
ask for forgiveness - so that "the spec was silent" and "the comment was
stale" (the two failures beneath this year's forty numeric bugs) stop
being discoverable only by a three-day adversarial audit.
