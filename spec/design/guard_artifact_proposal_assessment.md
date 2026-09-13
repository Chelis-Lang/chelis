# Guard-Artifact Proposal Assessment

**Status:** PROPOSED review guidance for chelis#1868, chelis#1869, and
chelis#1870. This document does not amend the agent contract, a numbered
specification, or an executable oracle. Each proposal remains open until its
own implementation and acceptance evidence land.

**Related decisions:** chelis#1824 decides where pull-request validation runs.
Chelis#1882 reports that the wire and bindings capacity censuses run in no
automated job, but that premise is stale on the assessed head: the daily and
manual heavy-E2E dtype job owns both censuses.

**Evidence snapshot:** the original review was written on 2026-09-12 from
`main` at `0e858f184` and later evidence through `72a1fb497`. This repository
copy was refreshed on 2026-09-13 against `main` at `23729c638`, including every
first-parent merge after the local report's final timestamp
(2026-09-12 11:07 EDT). The companion
[`guard_artifact_proposal_evidence.md`](guard_artifact_proposal_evidence.md)
records coverage, measurements, and known limits.

The refresh also corrects an execution-reach error in the local report and the
first repository draft. The heavy-E2E workspace shards exclude
`capacity_census_wire` and `capacity_census_bindings` because the dedicated
dtype Phase 0-3 oracle executes them. They remain absent from pull-request
`ci-fast`, but they are not local-only.

## Decision summary

| Proposal | Recommendation | Required correction or condition |
|---|---|---|
| chelis#1868, frozen digests | Adopt with changes, per artifact | Preserve the Phase 0 mutation implementation binding; replace growing-artifact hashes only with derived reports that retain the same review obligations and mutations. |
| chelis#1869, inventories | Adopt with one change | Keep positive `required` vocabulary evidence at crate granularity; do not replace reviewed classifications with regeneration. |
| chelis#1870, diagnostic pins | Adopt with changes | Use structured rejection identity for `unsupported` diagnostics, not a second prose parser; deliver the scheduled liveness canary before narrowing the pull-request check. |
| chelis#1824, pull-request execution reach | Decide first or in parallel | A correct guard outside the pull-request selection has a delayed failure mode. Do not retire an existing guard before its replacement has an executable owner at the required cadence. |
| chelis#1882, scheduled census reach | Refresh or close its stale premise | The daily/manual dtype Phase 0-3 job executes both censuses. The issue's broader demand for a citable automated verdict is implemented; per-pull-request reach belongs to chelis#1824. |

The governing distinction is:

> A guard list carries a review disposition per row or is regenerated from the
> tree. A hand-typed copy of what the tree contains is not a guard.

The frozen-digest corollary is:

> A digest is useful when it binds a reviewed claim that cannot be reconstructed
> from the current tree. A digest over ordinary tree growth is a review trigger,
> not semantic evidence, and may be replaced only by a derived trigger that
> preserves the same obligations.

## chelis#1868: frozen digests and freeze constants

The proposal covers eight artifact groups. They do not have one answer.

### Phase 0 runtime-representation freeze: split, but retain mutations

Keep `foundation_rows` and `source_inventory.mutations` inside the frozen
object. Each mutation row binds a witness identity, exact implementation
digest, expected failure, and command. The executed expected-failure contract
cannot detect a witness rewritten to fail trivially for the expected reason.

Drop `release_reproducers`, hardware probes, registered-source counts, and
other ordinary oracle configuration from the digest. Those entries are
reviewable directly as code and account for most routine freeze movement.

This is not theoretical. chelis#1828 changed the implementation digest of 16
of 44 existing mutation witnesses without adding or removing a witness. Every
mutation still ran and passed. Only the implementation binding could reveal
that what the witnesses checked had changed.

### Phase 1 and Phase 2 selection manifests: use superset semantics

Reject chelis#1868's original regenerate-on-failure replacement. It would
remove the blocking check on deletion of a frozen test.

Adopt chelis#1869's superset form:

- every frozen identity remains listed;
- every listed identity is non-ignored, selected, executed, and passing;
- additions are reported as review content rather than failed;
- removals and renames remain blocking changes.

Under this rule, the frozen selection is stable in the sense that matters:
ordinary additions do not require a digest move, while loss of reviewed
coverage still fails.

The current-main extended run supplies an exact witness. Runtime
Representation Phase 2 expected 237 tests in its broad shared-indexing leg and
failed selection equality after
`chelis-backend-c::host_abi_tests::fixed_control_host_helper_uses_the_active_invocation_rng`
entered that command. An exact `nextest list` comparison found that one
additional passing test and no removed frozen identity. The test arrived in
chelis#1826 before the local assessment, so this was a missed live instance,
not a hypothetical replacement benefit.

### Phase 3 required-test definition digests: replace, preserving the review cue

Retire per-definition hashes only after a derived report can:

- name every changed required test against the merge base;
- derive `parity_corpus_is_complete` from executable examples rather than
  preserving another hand-typed tree copy;
- require each protected test body to call its owning comparator or receipt
  entrypoint;
- run the existing empty-body, removed-row, and forged-receipt mutations; and
- print the existing doctrine that changing the guard merely to accept an edit
  is not a repair.

Post-review evidence from chelis#1918 sharpens this condition. The pull request
re-froze `parity_corpus_is_complete` after four filenames had accumulated:
three previously reviewed additions and its new
`annotated_concat_softmax.ch`. The old digest had therefore lagged three
merged changes. At the same time, the reviewer used the digest movement to
perform an explicit four-name freeze audit and verify removal and empty-body
mutations.

That audit is real review value. The corrected conclusion is not that the hash
provided no value; it is that a derived changed-test report can provide the
same cue without allowing a stale hash to certify a stale definition. The
replacement must make that review step unavoidable and durable.

### Phase 4B atom and region digests: replace only with an owning report

Retire `FROZEN_ATOM_DIGESTS` and `FROZEN_REGION_DIGESTS` only if the replacement:

- derives changed atoms and regions from the merge base;
- includes normative registry text in its owning atom;
- feeds an atom- or region-granular `Frozen-contract-change:` acknowledgement,
  rather than printing an informational CI line;
- preserves required-literal anchors and their mutation tests; and
- carries the existing doctrine: a changed atom owes its owning spec or design
  update, every consuming contract, and an adversarial mutation.

The hash value is mechanical. The change is not. Since chelis#1495, 20 of 25
atom or region digest moves added required-literal anchors or registry headings
to the oracle, while the other five edited the covered region itself. The
derived report must retain the trigger that puts those obligations in front of
the author and reviewer.

### Closed-corpus and count ratchets

- Keep `FROZEN_PHASE_A_DIGEST` while Phase A remains closed. Its rare moves
  have carried measured row-by-row deltas.
- Keep the temporary Nix workflow digest only while its event-policy condition
  remains explicit; give it a retirement condition.
- Keep the loud-unsupported `BASELINE`. It is a shrink-only count census, not a
  content hash, and it has recorded true catches.

Five post-review runtime-extent changes updated the Phase B derived target
manifest in their owning feature pull requests without a later repair-only
merge. None moved the closed Phase A digest. That is fresh evidence for the
finished-artifact distinction.

## chelis#1869: inventories, censuses, and manifests

### Derive existence; retain reviewed classification

Keep censuses, registries, and manifests where each row carries a disposition
a machine cannot invent: nonnumeric, tagged transport, exact numeric operation,
normative atom, accepted debt owner, or another reviewed authority class.
Regeneration must never bless an unclassified numeric surface.

chelis#1927 is a current example. A capacity-census fixture became illegal
after module-export enforcement changed. The repair corrected the fixture and
its owning explanation while preserving the classifier, baseline, inventory,
comparison, and denominator. That is the right repair shape for a
judgement-bearing guard.

### Replace the two expand inventories

The compiler checker owns the semantic property: rank-raising use must not
survive under the `expand` spelling. The inventory scanners count textual
occurrences and attach prose reasons; a count cannot decide whether an
occurrence is rank-preserving.

The post-review sequence is direct evidence:

1. chelis#1894 added a reviewed row for three call sites.
2. chelis#1914 changed the runtime-extent row from 27 to 32 and described five
   new named-bystander spellings.
3. The same pull request retained two additional valid broadcast controls that
   the row did not count.
4. chelis#1936 repaired 32 to 34 after the owning test failed during unrelated
   chelis#1926 integration work.

The scanner itself is sensitive: chelis#1936's review proved that one added or
removed call changes the count and that comments do not. What it cannot prove
is the semantic classification written beside the count. The repair therefore
strengthens the replacement case rather than showing that the list owns the
language property.

Do not delete the inventories first. Land the checker-backed or derived
replacement beside them, demonstrate the existing added/missing mutations,
then remove the hand copy.

### Keep crate-granular positive vocabulary evidence

Do not drop `required` markers from
`closed_vocabulary_architecture.rs`. Make them crate-granular so an owner may
move within the consumer crate without failing a file-path pin.

The `forbidden` list names only spellings already imagined. It cannot prove
that some typed path is used at all. Positive crate-level evidence directly
addresses the risk that an orthogonal implementation bypasses the tagged
carrier or typed vocabulary.

### Header censuses need hermetic closure

Adopt the Phase 0 scanner's closure contract for backend header lanes:
freestanding preprocessing, committed SDK fixtures, complete attributed
include closure, and a planted include-outside-universe mutation.

Do not add a Metal SDK stub merely to satisfy the current lane. A header made
only of `static inline` functions produces no exported ABI row; the lane must
first have an attributable public declaration to census.

### Keep the derived runtime-extent target manifest

The runtime-extent target JSON changed in chelis#1873, chelis#1909,
chelis#1914, chelis#1911, and chelis#1943. Each feature pull request changed
the oracle, baseline, target manifest, and owning design together. No later
target-manifest repair was required.

This is the successful comparator for the expand inventories: derived
selection plus a current tripwire, rather than a count copied beside the tree.

### `.config/ci-test-targets.toml` remains chelis#1824's decision

The file grew from 53 to 61 targets after the original assessment:

- three targets in chelis#1894;
- three targets in chelis#1918; and
- two targets in chelis#1926.

All eight were added by the pull request introducing or depending on those
tests. That is useful immediate protection, but still reactive growth without
an admission rule. chelis#1824 must decide default inclusion, exclusions,
change detection, and cost ownership. This assessment does not prescribe a
longer hand-maintained list.

## chelis#1870: diagnostic pins and rejection identity

### Scope the identity rule to `unsupported` diagnostics

For `unsupported` diagnostics, the blocking contract is structured identity:

- the `unsupported:` brand;
- the named subject and its context;
- the earliest competent compiler stage;
- `DiagnosticKind::UnsupportedFeature`, rendered as the machine-facing
  `unsupported_feature` kind;
- the span's presence and association where one exists;
- the correct supported alternative where one exists;
- the typed disposition category and its deciding numbered atom or exact
  capability cell; and
- explicit human review that any associated issue is relevant, not merely
  structurally valid and open.

Project issue metadata may schedule work around a capability cell, but it is
not semantic authority. Hint wording, wrapper prefixes, node identifiers, and
the exact textual rendering of offsets are snapshot evidence; they do not
replace the blocking span and authority fields.

Expose the complete `Unsupported` value through a structured channel or keep
the existing substring and exact-text identity locks and their mutations until
that channel covers every obligation above. Do not ship a second hand-written
parser for the rendered sentence.

This is not a rule against every exact diagnostic assertion. Post-review
changes chelis#1873, chelis#1926, and chelis#1943 deliberately lock exact public
runtime-error transport and cross-lane rendering. Their owning contracts make
the bytes observable. chelis#1870 addresses unsupported prose whose numbered
contract identifies structure rather than exact wording.

### Include the new chelis#1918 pins

chelis#1918 added the same unsupported-softmax rendering to two test surfaces:
the shipped-example refusal census and a parity test that asserts complete
stderr. These are new hand-maintained prose copies after the proposal's
inventory.

Before implementation, classify the requirement:

- if complete CLI wording is intentionally part of that executable example's
  contract, keep one derived snapshot and share its identity;
- otherwise make structured rejection identity blocking and leave prose as the
  reviewed snapshot.

Either way, chelis#1870's migration inventory must include both sites.

### Keep working identity guards

- Keep Phase 3 dropout rows, stated per target and entry path rather than as an
  unbounded universal claim.
- Keep compile-time rejection-authority membership.
- Keep the loud-unsupported shrink-only count tripwire.
- Use one source for narrow-float capability prose, then point users to the
  normative dtype matrix rather than restating its contents.

### Deliver the canary before narrowing liveness

The §C7.5 scheduled workflow still does not exist on `main`. The pull-request
job still validates the standing issue manifest. Therefore the current job is
still the only automated monitor for a cited issue that closes.

Land and validate both of these before narrowing the pull-request job:

1. a source-derived manifest that cannot retain an uncited row; and
2. the scheduled standing-liveness canary, including its workflow-registry and
   trigger-loop controls.

Then restrict the pull-request check to rows the change adds or modifies.

## Deterrence and execution reach

Deterrence is recorded where a guard failure demands a judgement an author
cannot make merely to obtain green CI: capacity censuses, semantic registries,
the Phase 0 mutation inventory, positive vocabulary evidence, and identity
locks. Those mechanisms remain.

The corrected claim for recompute-style guards is narrower:

- no evidence shows that a content hash or copied count prevented an
  orthogonal implementation;
- chelis#1918 does show that a digest movement can prompt a careful review;
- that cue must survive in a derived replacement; and
- repeated mechanical re-freezes and count repairs train authors to treat guard
  files as routine bookkeeping, which can obscure the guards that require
  judgement.

Pull-request execution reach remains the largest immediate gap:

| Guard | Pull-request reach and delayed owner |
|---|---|
| `expand_call_site_inventory` | Not selected by `ci-fast`; reached by the daily/manual full-workspace and generalization jobs |
| `expand_insert_source_literal_inventory` | Selected by `ci-fast` and reached by the daily/manual broad jobs |
| rejected-cells corpus | Selected by `ci-fast` and reached by the daily/manual broad jobs |
| Phase 3 gate contract | Selected by `ci-fast` and reached by the daily/manual broad jobs |
| wire and bindings capacity censuses | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 job |
| §C7.5 standing-liveness canary | Workflow absent |

The contrast between the two expand inventories is useful but not a complete
mechanism verdict. The selected insert inventory remained synchronized through
chelis#1910 and chelis#1926; the unselected call-site inventory drifted and
needed chelis#1936. Distribution determines whether a defect is loud. It does
not turn a copied count into semantic authority.

## Post-review merge audit

Every first-parent merge after the local assessment's final timestamp was
classified:

| Pull request | Relevant effect |
|---|---|
| chelis#1873 | Updated the runtime-extent derived target manifest in the owning feature change; added an exact runtime-error contract outside chelis#1870's unsupported scope. |
| chelis#1894 | Added three CI targets and one reviewed expand-inventory row. |
| chelis#1909 | Updated the runtime-extent derived target manifest in the owning feature change. |
| chelis#1913 | Corrected an investigation document; no guard-mechanism effect. |
| chelis#1918 | Re-froze the Phase 3 parity definition after four additions, added three CI targets, and added two exact unsupported-softmax prose copies. |
| chelis#1914 | Updated the runtime-extent manifest and the expand count, but omitted two retained occurrences later repaired by chelis#1936. |
| chelis#1927 | Repaired a capacity-census fixture without weakening its classifier or reviewed inventory. |
| chelis#1910 | Updated a legitimate insert-inventory count in the owning change. |
| chelis#1926 | Updated a legitimate insert-inventory count, added two CI targets, and documented exact fatal diagnostic transport outside chelis#1870's unsupported scope. |
| chelis#1911 | Updated the runtime-extent derived target manifest in the owning feature change. |
| chelis#1936 | Repaired the post-chelis#1914 expand count from 32 to 34 and proved the scanner's added/missing sensitivity. |
| chelis#1943 | Updated the runtime-extent derived target manifest in the owning feature change; added an exact lowering diagnostic contract outside chelis#1870's unsupported scope. |

None implements or closes chelis#1868, chelis#1869, or chelis#1870.

## Current-main extended validation

Workflow-dispatched Linux Extended Validation run `34728303804` completed on
the exact assessed head, `23729c638098f40e72b9a9834c6338318c41a5ba`.
Across 20 jobs, six succeeded, 13 failed, and one telemetry job was skipped
because its prerequisites were red. The red run does not overturn the three
artifact recommendations, but it adds four material findings.

First, the Runtime Representation Phase 2 selection failure is the live
positive-addition witness described above. Equality rejected one new passing
unit test without losing a frozen identity. That directly strengthens
chelis#1868's superset replacement.

Second, the dtype Phase 0-3 job selected and executed both capacity census
binaries. Its flattened union ran 772 tests: 769 passed and three failed. The
wire census's full current-baseline test passed after 968 seconds. The bindings
census reached its full 1,246-second test and failed because the current native
execution worker triggers GCC's `-Wmisleading-indentation` error tracked by
chelis#1864. Thus chelis#1882's local-only premise is false even though the
current automated verdict is red.

Third, the dtype job's other two failures were the missing Metal SDK fixture
tracked by chelis#1866. That is fresh executable support for chelis#1869's
hermetic attributed-closure recommendation, not a reason to weaken or bypass
the census.

Fourth, chelis#1918 added `annotated_concat_softmax.ch` to the audited frozen
parity definition, but this run found that the same example fails both strict
Deep validation and `--desugar` corpus acceptance. The digest movement
successfully prompted a filename audit; it did not prove the example's broader
executable obligations. The replacement report must therefore join changed
names to their owning executable checks, not reproduce only the name review.

Compared with scheduled run `34670561943` on `6abca2406`, eleven distinct
tests that previously passed are now failing. Ten had no exact open owner:

- four chained-expand evaluation or C-agreement rows fail on a conflicting
  synthetic dimension;
- the chelis#345 gradient row rejects backward-DAG construction;
- three proof or opaque-type fixtures now encounter module-export rejection
  before their expected contract; and
- the two `annotated_concat_softmax.ch` corpus checks fail.

The eleventh,
`runtime_extent_claim_preparation::omitted_extent_claim_contract`, is already
tracked by chelis#1917.

No exact open issue match was found for those ten rows as of 2026-09-13. This
document records them as current-main residual findings rather than assigning
an unproved root cause or expanding this documentation pull request into their
repair.

The remaining failures reconfirm already-open work, including chelis#1746,
chelis#1776, chelis#1779, chelis#1784, chelis#1842, chelis#1861 through
chelis#1864, chelis#1866, chelis#1881, and chelis#1917. They do not change the
mechanism conclusions here. In particular, the rejected-cells corpus passed
in the dtype union, while the §C7.5 standing-liveness canary remains absent;
chelis#1870's structured-identity and sequencing recommendation is unchanged.

## Review and delivery order

1. Adopt chelis#1869's disposition-versus-regeneration rule as the governing
   design sentence; treat the digest rule as its corollary.
2. Give the retained guards executable owners at their required cadence. Treat
   chelis#1882's scheduled-owner premise as satisfied by the dtype Phase 0-3
   job, and decide chelis#1824 for pull-request selection rather than adding
   targets indefinitely.
3. Trial replacements beside existing guards:
   - derive Phase 4B changed atoms and regions while retaining the digest;
   - derive the Phase 3 changed-test set while retaining definition hashes;
   - add structured unsupported identity checks while retaining snapshots; and
   - measure Phase 1 superset behavior without changing acceptance.
4. Require every replacement to fail on the planted mutation that justified
   the old guard.
5. Deliver the §C7.5 canary, then narrow liveness.
6. Retire old artifacts only after the parallel trial records equivalent
   detection and review output.

Record the three proposal implementations as one coordinated decision even if
delivery is split into separate pull requests. The interactions are semantic:
one slice decides what is derived, another what is reviewed, and the third what
is blocking identity versus rendered evidence.

## Non-claims and open evidence

This assessment does not establish:

- how often local `--fast` tripwires fire when no pull-request text records it;
- whether reviewers re-read every changed atom when a digest moves;
- whether required-literal anchors would continue to grow without the current
  digest round trip;
- whether the primary backend-header census shares the host-SDK exposure found
  in the device lanes; or
- the final cost and pull-request scheduling policy chelis#1824 must place.

Those unknowns are reasons to run parallel trials, not reasons to silently
remove or indefinitely retain the current artifacts.
