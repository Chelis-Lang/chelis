# Guard-Artifact Proposal Evidence and Coverage

**Status:** Evidence appendix for
[`guard_artifact_proposal_assessment.md`](guard_artifact_proposal_assessment.md).
It records what was inspected, the measurements that changed a verdict, the
post-review refresh, and what remains inferred. It is not normative authority
and does not by itself authorize an oracle change.

**Tracking:** chelis#1868, chelis#1869, chelis#1870, and chelis#1824.
Chelis#1882 was closed with the citable heavy-E2E census executions below.

## Review method

The original review used:

- the full bodies and comments of the three proposals and their cited issues;
- the three family reports for digests, inventories, and diagnostic pins;
- the guard-C collect-all and identity mutation probe;
- first-parent history for digest and inventory movement;
- merged pull-request bodies searched for evidence of deterrence;
- the current implementations of each proposed replacement seam; and
- the owning design rationale for the high-risk artifacts.

A second pass read the originating rationale for the Phase 0 freeze, Phase 3
definition digests, and Phase 4B atom and region digests, then ran recurrence
searches for the census and substitution-count ratchets.

The first 2026-09-13 refresh classified every first-parent merge after the
local report's final timestamp and inspected the changed artifacts, owning
pull-request threads, and CI reach through `main` at `23729c638`. A second
refresh through `main` at `8d1293f31` classified the implementation sequence
through chelis#2008 and re-measured the current PR integration surface.
The CI implementation refresh re-measures that surface against `b38efcfbe`.
The delivery refresh below uses merged `main` at `929b2c483` after chelis#2053;
it preserves the earlier measurements under their original commits.

## Per-artifact rationale coverage

| Artifact | Owning rationale inspected | Assessment basis | Current conclusion |
|---|---|---|---|
| Phase 0 `FREEZE_SHA256` | `runtime_representation.md` §B1 and `coverage_manifest()` | Parsed freeze movement and mutation identity changes | chelis#1969 split the freeze; chelis#2017 binds the complete reviewed mutation-rejection contract. Ordinary release-leg configuration remains code-derived. |
| Phase 1/2 `MANIFEST_SHA256` | Selection code and proposal history | Equality behavior, chelis#1817 drift, interaction with removal | Implemented by chelis#1959: additions report and execute; removals fail. |
| Phase 3 required-test digests | `faithful_observation.md` Phase 3 rationale and oracle mutations | Originating failure classes, chelis#1724, chelis#1818, chelis#1918 | Retired in chelis#2053 after corpus derivation, mandatory changed-test review, and standing body/result controls. The finite mutation set is covered; arbitrary test semantics are not proved. |
| Phase 4B atom/region digests | Oracle doctrine, dtype amendment ledger, post-chelis#1495 moves | Per-move anchor and region classification | Retired in chelis#2045 after granular reporting and acknowledgements; identities, anchors, boundaries and semantic controls remain. |
| Phase A digest | Runtime-extent design and move receipts | Closed corpus, measured moves | Keep while Phase A remains closed. |
| Nix workflow digest | Test docstring and issue history | Temporary event policy, no movement | Keep with an explicit retirement condition. |
| Loud-unsupported count baseline | Loud-unsupported rationale and chelis#1348 | Shrink-only census and true catches | Keep. |
| Expand call/insert inventories | Headers, checker owner, history, chelis#1936 mutations | Scanner sensitivity versus semantic classification | Retired by chelis#1968 after checker-backed replacement; stale PR selection removed by chelis#1998. |
| Closed-vocabulary markers | Complete test header/tables and known evasions | Positive use versus finite forbidden spellings | Crate-granular positive evidence implemented by chelis#1960. |
| Capacity census baselines and registries | Agent contract, census implementation, representative review history | Rows carry authority a machine cannot invent | Keep; improve executable reach and hermetic closure. |
| Backend header lanes | Phase 0 closure precedent and device-lane preprocessing probes | Host SDK leakage and zero-row Metal lane | Complete attributed discovery and SDK fixtures implemented by chelis#1963. |
| Runtime-extent target manifest | Manifest code, tripwire, post-review changes | Five synchronized feature changes, no repair-only drift | Keep. |
| `.config/ci-test-targets.toml` | chelis#1824 and current history | Historical 73-row base, 779 eligible targets, and 36 duplicate names across packages | chelis#2019 delivered required added/directly-modified target execution and both standing expand replacements; chelis#2051 added the standing body audit. Broader expansion remains informational. |
| Rejected-cells corpus | File contract, unsupported structure, collect-all and mutation probe | Byte drift versus stable identity, cross-lane skew | The in-process structured trial and typed lowering kind landed; exact-text locks remain until complete authority and serialized identity can replace them. |
| Dropout gate rows | Gate implementation and entry-path findings | Real behavior changes and policy-fork detection | Keep, with per-target and per-entry claims. |
| Rejection-authority liveness | Design §C2.1/§C7.5, workflow, validator | Useful audits, bystander failures, scheduled-owner sequencing | Source-derived membership landed in chelis#1961 and the standing scheduled check in chelis#1971. The changed-row PR experiment from chelis#2025 was retired after repeated 10-14 minute runs selected no changed authority; `script-unit` retains privacy/source-usage checks and the always-running Docs Phase 4B oracle retains fresh compiler closure/generated agreement. |
| Narrow-float messages | Dtype matrix, compiler copies, chelis#1871 | Triplicated prose disagrees with normative source | Shared capability identity implemented by chelis#1972. |

## Measurements that changed or constrained verdicts

### Phase 0 mutation implementation binding

The first count of chelis#1828's changed
`implementation_sha256` entries was incorrectly reported as 32 because every
in-place JSON modification appears once as a removed diff line and once as an
added diff line.

Re-measurement keyed on `witness_id` found:

- 44 mutation entries before and after;
- zero added;
- zero removed;
- exactly 16 changed `implementation_sha256` values; and
- no other changed field in those entries.

An independent comparison initially reported zero because it keyed entries on
a truncated JSON prefix ending before the digest field. Both failures support
the same method rule: parse the artifact by stable identity; do not count its
diff or key on a textual truncation.

This measurement moved the Phase 0 verdict. The execution contract still
passed after 16 witness implementations changed, so it is not a substitute for
binding the implementation to the reviewed witness claim.

### Phase 4B digest movement

For the 25 atom or region digest moves after chelis#1495:

- 20 added required-literal anchors or registry headings to the oracle; and
- five edited the exact region covered by the moved digest.

The digest identifies a changed contract. The anchors and mutations defend it.
The acknowledgement failure text carries the same three stated obligations:
update the owning spec or design, update consuming contracts, and add an
adversarial mutation.

The derived replacement must preserve identification and put those obligations
in the author and review path. The evidence does not establish whether authors
would continue adding anchors if the current hash round trip disappeared.

### Phase 3 parity definition movement

chelis#1918 changed the frozen `parity_corpus_is_complete` definition by four
filenames:

- `dropout_entry.ch`;
- `dropout_static_rate.ch`;
- `wildcard_extents.ch`; and
- `annotated_concat_softmax.ch`.

The first three had already merged. The later freeze movement therefore did
not prevent their stale definition from residing on `main`.

The chelis#1918 reviewer nevertheless used the movement as a concrete review
cue: it compared the old and new definitions, proved the delta was exactly
those four names, and ran removal and empty-body controls. This is evidence for
preserving the review ritual, not for preserving the hash value.

### Expand inventory recurrence

The post-review call-site sequence was:

| Merge | Inventory effect |
|---|---|
| chelis#1894 | Added one reviewed row covering three occurrences. |
| chelis#1914 | Changed the runtime-extent row from 27 to 32 for five named-bystander spellings. |
| chelis#1936 | Corrected the same row from 32 to 34 for two retained same-rank controls omitted from the earlier count. |

chelis#1936 proved the scanner rejects one added occurrence, one removed
occurrence, and both removed occurrences, while ignoring Rust comments. That
is evidence that the scanner accurately counts the textual domain it
recognizes.

It does not prove the row's semantic reason. Both added occurrences had to be
read and classified as rank-preserving. That classification is the property
the checker should own.

The call-site inventory was not selected by `ci-fast`; the drift was found
during unrelated integration testing. The insert-source inventory was selected
and remained synchronized through chelis#1910 and chelis#1926. This supports
the execution-reach finding without converting either count into semantic
authority.

### Frozen-selection addition on current main

The workflow-dispatched extended run on `23729c638` failed Runtime
Representation Phase 2 at the final broad selection leg:

```
checked C shared indexing dtype dispatch and storage reuse:
list and execute 237 frozen tests
RUNTIME REPRESENTATION PHASE 2:
FAIL: empty, duplicate, or drifted frozen test selection
```

An exact local `nextest list` comparison against
`runtime_representation_phase1_tests.json` found one actual-only identity and
no expected-only identity:

```
chelis-backend-c::host_abi_tests::fixed_control_host_helper_uses_the_active_invocation_rng
```

That test entered `main` in chelis#1826, before the local assessment. The
failure is therefore a missed current instance of the proposal's stated
problem: an ordinary positive test addition invalidates an equality-frozen
selection even though no reviewed test disappeared. It directly supports
superset semantics, while leaving deletion and rename blocking.

### CI target-list growth

At the local report cutoff, `.config/ci-test-targets.toml` contained 53 target
entries. At `23729c638` it contained 61. At `8d1293f31` it contained 68.
At `b38efcfbe` it contains 73; the candidate adds the two replacement controls
for 75 standing identities.

| Merge | Added targets |
|---|---:|
| chelis#1894 | 3 |
| chelis#1918 | 3 |
| chelis#1926 | 2 |
| Later caller/formal and gradient-result scope | 4 |
| Wildcard-absorption coverage | 1 |
| Pipe-fold placement and surface coverage | 2 |
| Symbolic-reshape regression | 1 |
| Retired insert inventory row | -1 |

The additions are legitimate, but each came from the pull request currently
needing the test. No rule derives omissions or states which future test target
must be selected.

Cargo metadata at `b38efcfbe` contains 779 default-eligible integration targets in a
34-package workspace; 24 packages currently own at least one. Thirty-six
target names, covering 78 identities, occur in more than one package. The
base standing file covers 73 exact package/target identities. Those
measurements rule out global target-name uniqueness as an execution
requirement. The candidate uses package-qualified execution in both lanes.
The measurements also justify treating broader package expansion as a measured
heuristic rather than an impacted-test completeness claim.

Jeff Smith's incoming-main audit on chelis#1824 identified a concrete ownership
case after chelis#1968/chelis#1998. The retired
`expand_call_site_inventory` and `expand_insert_inventory` rows are correctly
gone, but at that audit neither `chelis-types::expand_insert_dispatch_family` nor
`chelis-types::issue_1294_standard_lowerings` was in the standing manifest. The
first protects the replacement AST dispatch positions. The second protects the
behavioral expand/insert distinction and participates in the broader dtype
builtin atom closure, which is not an ordinary per-pull-request owner.

An exact local run on `da11a6fa3` executed both replacement targets together:
12 tests passed in 0.060 seconds of nextest execution. The selected policy adds
both exact package/target identities to the standing manifest. The candidate
contains both, taking the current base from 73 to 75 rows. This is a bounded
correction for controls that replace a previously selected guard class; it does not restore the copied
inventories or claim that every test in `chelis-types` belongs in standing CI.

## Executable reach before the CI implementation

Checked on `main` at `8d1293f31`:

| Artifact or test | Reach |
|---|---|
| checker-backed `expand` contract | Both copied inventories are retired; semantic coverage remains in ordinary package tests and daily/manual broad jobs |
| `chelis-types::expand_insert_dispatch_family` | Absent from current `ci-fast`; selected to enter the standing manifest as a small replacement AST-position control |
| `chelis-types::issue_1294_standard_lowerings` | Absent from current `ci-fast`; reached by the broader dtype builtin atom closure and selected to enter the standing manifest for ordinary PR ownership |
| `issue_687_rejected_cells_corpus` | Present in `.config/ci-test-targets.toml` and reached by daily/manual broad jobs |
| `phase3_gate_contract` | Present in `.config/ci-test-targets.toml` and reached by daily/manual broad jobs |
| `capacity_census_wire` | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 oracle |
| `capacity_census_bindings` | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 oracle |
| §C7.5 `loud-unsupported-nightly.yml` | Daily/manual standing-liveness owner on `main`; first default-branch run `34765689930` passed on `da11a6fa3` in 10m45s without opening a failure issue |
| current rejection-issue validator | Runs in `ci.yml` against the standing manifest |
| new/changed integration target outside the current standing manifest | Automatically reached by full Linux nightly; omitted from regular PR execution until the exact change-owned lane lands |

The generic heavy-E2E workspace and generalization commands exclude both
capacity binaries, but `.config/nextest.toml` delegates them to the dtype
profile and `scripts/dtype_phase1_oracle.py` selects both. The heavy-E2E dtype
Phase 0-3 job flattens that inherited selection and runs it with current
framework-owned receipts. The exclusion is deduplication, not absence.

The strongest retained guards still have uneven pull-request ownership. A
guard that runs only after merge is a delayed signal to the pull request, but
it is not the same condition as a local-only guard. Chelis#1882's stated
requirement that the census be reachable from some citable automated job is
therefore met on this head; chelis#1824 still owns the per-pull-request policy.

## Delivered execution and measured limits

Chelis#2019 superseded the pre-implementation PR omissions above. Both expand
replacement targets now have standing coverage, and added/directly-modified
default-enabled integration targets receive required package-qualified
execution. Chelis#2051 added the standing per-body audit; chelis#2053 extended
its checked-result roles before retiring the 51 definition hashes. Required
identities, runtime receipts, comparator canaries and changed-test review
acknowledgements remain.

The delivery evidence distinguishes required execution from the separate
package-expansion trial:

| Change and commit | Hosted evidence | Meaning |
|---|---|---|
| chelis#2019, `96f8719362` | [CI 34776588597](https://github.com/Chelis-Lang/chelis/actions/runs/34776588597), [Hull 34776588782](https://github.com/Chelis-Lang/chelis/actions/runs/34776588782) passed | Required change-owned delivery and package-qualified standing execution landed. |
| chelis#2040, `8fba69793` | [CI 34810301530](https://github.com/Chelis-Lang/chelis/actions/runs/34810301530): required jobs passed; three expansion workers timed out, one returned a failing receipt | The 780-target shared-path expansion does not fit the selected budget; feature-empty targets and HIP failures also need their proper owners. |
| chelis#2051, `529bd6705` | [CI 34815817158](https://github.com/Chelis-Lang/chelis/actions/runs/34815817158): required jobs passed; four optional workers were cancelled by the next main push | No expansion receipts were produced. This is cancellation, separate from the candidate run's four explicit 20-minute timeouts. |
| chelis#2052, `dc3b76368` | [CI 34817567461](https://github.com/Chelis-Lang/chelis/actions/runs/34817567461), [Hull 34817567414](https://github.com/Chelis-Lang/chelis/actions/runs/34817567414) passed | All eight shard receipts validate zero dynamic targets for this script-only repair; 78 focused controls and the real binding liveness command passed separately on the merged commit. |
| chelis#2053, `929b2c483` | Two fresh reviews and required [candidate CI 34818246419](https://github.com/Chelis-Lang/chelis/actions/runs/34818246419) passed on `39a61e730`; its inspected and hosted merge tree equals the actual merge | The required body-audit target and all four receipts passed. The separate 285-target expansion does not certify the retirement or a complete package run. |

The per-PR records carry exact reviewed heads, review findings and repairs,
local acceptance commands, and receipt verification. A green report that
records missing or failing optional shards is not a green package test run.
These observations support retaining informational expansion while chelis#1824
resolves its cost, prerequisite and standing-admission decisions; they do not
authorize an exclusion or a weaker required guarantee.

The selected follow-up uses that evidence to remove automatic package expansion
from PR updates and main pushes while retaining the nightly backstop. Agents
dispatch expansion once on the reviewed final PR head through a workflow that
accepts the PR number and expected head SHA and rejects stale candidates.
Acknowledgement enforcement moves to a separate required check so description
edits do not enter compiler or Hull workflows and therefore cannot cancel or
replace implementation evidence. Base retargets remain implementation events:
a separate required head receipt waits for trusted-base exact-head/exact-base
CI and Hull dispatches against the new synthetic merge. The separate
execution-optimization candidate adds
plan-proven empty-shard receipts before setup, exact
candidate/configuration-bound standing coverage reuse, and package-batched
expansion with exact target/test reconstruction. Its local controls cover empty
selections, duplicate target names, stale or tampered overlap, missing results,
failures, timeouts and retained partial results. Comparative hosted cost
measurements remain required and are not established by the runs above.

## Historical extended-run evidence at `23729c638`

Linux Extended Validation run `34728303804` was manually dispatched on
2026-09-13 and completed against
`23729c638098f40e72b9a9834c6338318c41a5ba`. Its final job accounting was:

- six successful jobs;
- 13 failed jobs; and
- one skipped telemetry job.

This is main-head evidence, not validation of the documentation pull request.
The failed broad jobs contain many repeated instances, so this appendix
classifies distinct failure classes rather than treating every shard copy as a
different defect.

### Evidence that changes or sharpens this assessment

| Evidence | Executed result | Assessment effect |
|---|---|---|
| Runtime Representation Phase 2 frozen selection | Expected 237 identities; exact list comparison found one additional `fixed_control_host_helper_uses_the_active_invocation_rng` test and no removed identity | Direct positive-addition witness for chelis#1868's superset semantics |
| Dtype Phase 0-3 selection | The flattened command explicitly selected `capacity_census_wire` and `capacity_census_bindings` | Corrects the local report and chelis#1882 local-only premise |
| Wire full census | `wire_schema_numeric_fields_match_the_reviewed_baseline` passed in 968.042 seconds | Supplies a citable current automated wire verdict |
| Bindings full census | `registered_pyfunctions_match_the_reviewed_rustdoc_signatures` ran for 1,246.026 seconds, then failed through chelis#1864's native C fixture warning | Proves scheduled ownership while keeping the current verdict honestly red |
| Backend-header census | Two exact Metal rows failed because `Metal/Metal.h` was unavailable | Reconfirms chelis#1866 and strengthens chelis#1869's hermetic-closure condition |
| Phase 3 parity example | `annotated_concat_softmax.ch` failed both strict Deep validation and `--desugar` acceptance after chelis#1918 froze its name | Shows that the digest's review cue is not executable acceptance evidence |
| Rejected-cells corpus | Passed in the current dtype union after failing in the preceding scheduled run | No new contradiction to chelis#1870; exact prose remains working evidence, not the proposed long-term blocking identity |

The dtype union selected 772 tests: 769 passed and three failed. The failures
were the two chelis#1866 Metal rows and the chelis#1864 bindings-census worker.
This is stronger reach evidence than source inspection alone because both
census entrypoints and their slow full checks actually ran.

### Newly failing rows relative to the preceding scheduled head

Scheduled run `34670561943` on `6abca2406` passed each of the following tests.
The current run failed eleven distinct rows. Ten had no exact open owner:

| Failure class | Distinct rows | Current observation |
|---|---:|---|
| Chained expand | 4 | Host evaluation or C agreement panics because a synthetic dimension resolves to conflicting actuals. |
| chelis#345 gradient | 1 | Evaluation rejects construction of the backward DAG at same-shape/same-dtype `relu` verification. |
| Module-export fixture precedence | 3 | One proof fixture and two opaque-type fixtures encounter an unexported import before the result their assertions expect. |
| `annotated_concat_softmax.ch` corpus validity | 2 | Strict Deep validation and `--desugar` reject the executable example added by chelis#1918. |

The eleventh row,
`chelis-cli::runtime_extent_claim_preparation::omitted_extent_claim_contract`,
is already tracked by chelis#1917.

Exact open-issue searches on 2026-09-13 found no matching owner for these ten
rows. That absence is recorded as an assessment limit: the evidence proves the
current failures and their previous-head passes, but this review did not bisect
or assign their production causes.

### Reconfirmed tracked failures

The rest of the red jobs repeat already-open classes:

- chelis#1861 through chelis#1863 for HIP vocabulary, dispatch, runtime-library,
  and header prerequisites;
- chelis#1864 for `PRIu64` and GCC warning portability;
- chelis#1866 for the non-hermetic Metal header lane;
- chelis#1746, chelis#1776, chelis#1779, chelis#1784, chelis#1842,
  chelis#1881, and chelis#1917 for their existing compiler, corpus, backend, or
  runtime-extent rows.

Those failures matter to main's state but do not select between the guard
replacement mechanisms assessed here.

## Deterrence evidence

Recorded deterrence exists for guards whose repair requires judgement:

- chelis#1149 stopped on a capacity-census decision and escalated;
- chelis#1518 selected an additive design because a successor row could not be
  silently dispositioned;
- chelis#1848 declined multiple routes rather than edit census guard files;
- chelis#1882 records four wire-census refusals encountered by hand before its
  current scheduled owner was recognized;
- chelis#746 found an uncensused stub on introduction; and
- chelis#1474 retired rows only with successor registrations and mutations.

The post-review chelis#1927 repair follows the same pattern: correct an illegal
fixture and its explanation while leaving classifier, baseline, inventory, and
comparison intact.

For recompute-style guards, the evidence is different:

- no recorded hash move caused an orthogonal implementation to be redesigned;
- chelis#1724 narrowed coverage after a digest blocked a corpus addition;
- chelis#1818 and chelis#1918 show definitions can remain stale across merges
  before a later re-freeze; and
- chelis#1918 also shows that the movement can trigger a real audit.

The defensible conclusion is therefore not “hashes have no effect.” It is:
hashes are review triggers whose useful effect must be reproduced explicitly;
they are not themselves evidence that the current tree satisfies the reviewed
claim.

## Recurrence checks for retained ratchets

### Capacity census

Within the enumerated primary roots, no new bare numeric carrier was found
after the ratchet. The open chelis#1583 instance is in `chelis-e2e`, outside
the typed census roots by construction. That is a scope gap, not a failure
inside the declared domain.

### Loud-unsupported substitution count

The chelis#795 spelling is inside the ratchet's reach, baselined, and blocked
behind the checker guard. The chelis#958 path-qualified spelling remains a
known pre-existing evasion because the current tripwire does not recognize
path-qualified forms. No new in-reach occurrence was found.

These results support retaining the ratchets while separately tracking their
declared reach and known evasions.

## Exact-diagnostic scope after the review

Post-review merges added exact diagnostic checks for different reasons:

- chelis#1873 and chelis#1943 require cross-lane exact runtime/lowering error
  rendering for bounded extent claims;
- chelis#1926 requires exact public fatal diagnostic transport after helper
  probe cleanup; and
- chelis#1918 added exact unsupported-softmax stderr in both a shipped-example
  census and parity test; while
- chelis#1973 added typed unsupported identity through the compiler/API adapter
  boundary while retaining compatible rendered text and the reviewed exact
  snapshots.

The first two classes have explicit owning claims about exact observable
rendering. They are not evidence that all prose should become identity-only.

The chelis#1918 unsupported copies do fall within chelis#1870's design question
because the numbered unsupported contract identifies structured fields.
Chelis#1973 deliberately retained them as reviewed exact-text contracts while
adding typed identity beside the rendering. Later migrations may revisit those
sites, but the bounded trial did not silently weaken them.

## Evidence still missing

The review did not establish:

- local guard-firing frequency where pull-request bodies record only the final
  green result;
- complete review-thread behavior for every historical digest move;
- whether atom anchors continue to grow without the current oracle-file edit;
- steady-state standing-liveness cost beyond the first successful
  10-minute-45-second default-branch receipt;
- the exact cost distribution of optimized manual changed-package expansion on
  representative pull requests; or
- whether reviewed shared-path rules need broader package mappings.

Each implementation slice should close only the unknowns it needs. None of
these gaps justifies a broad “remove all ratchets” or “retain all ratchets”
decision.
