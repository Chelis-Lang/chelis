# Guard-Artifact Proposal Evidence and Coverage

**Status:** Evidence appendix for
[`guard_artifact_proposal_assessment.md`](guard_artifact_proposal_assessment.md).
It records what was inspected, the measurements that changed a verdict, the
post-review refresh, and what remains inferred. It is not normative authority
and does not by itself authorize an oracle change.

**Tracking:** chelis#1868, chelis#1869, chelis#1870. Related execution-reach
decision: chelis#1824. Chelis#1882's local-only premise is contradicted by the
current heavy-E2E dtype owner and should be refreshed separately.

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

The 2026-09-13 refresh classified every first-parent merge after the local
report's final timestamp and inspected the changed artifacts, owning pull
request threads, and current CI reach through `main` at `23729c638`.

## Per-artifact rationale coverage

| Artifact | Owning rationale inspected | Assessment basis | Current conclusion |
|---|---|---|---|
| Phase 0 `FREEZE_SHA256` | `runtime_representation.md` §B1 and `coverage_manifest()` | Parsed freeze movement and mutation identity changes | Freeze `foundation_rows` plus `source_inventory.mutations`; drop ordinary oracle configuration. |
| Phase 1/2 `MANIFEST_SHA256` | Selection code and proposal history | Equality behavior, chelis#1817 drift, interaction with removal | Use superset semantics; additions report, removals fail. |
| Phase 3 required-test digests | `faithful_observation.md` Phase 3 rationale and oracle mutations | Originating failure classes, chelis#1724, chelis#1818, chelis#1918 | Replace only with a derived changed-test report preserving comparator and mutation obligations. |
| Phase 4B atom/region digests | Oracle doctrine, dtype amendment ledger, post-chelis#1495 moves | Per-move anchor and region classification | Replace only with atom-granular derived reporting, acknowledgements, anchors, and doctrine. |
| Phase A digest | Runtime-extent design and move receipts | Closed corpus, measured moves | Keep while Phase A remains closed. |
| Nix workflow digest | Test docstring and issue history | Temporary event policy, no movement | Keep with an explicit retirement condition. |
| Loud-unsupported count baseline | Loud-unsupported rationale and chelis#1348 | Shrink-only census and true catches | Keep. |
| Expand call/insert inventories | Headers, checker owner, history, chelis#1936 mutations | Scanner sensitivity versus semantic classification | Replace after checker-backed or derived coverage runs beside them. |
| Closed-vocabulary markers | Complete test header/tables and known evasions | Positive use versus finite forbidden spellings | Keep `required` at crate granularity; keep and widen structural forbidden detection. |
| Capacity census baselines and registries | Agent contract, census implementation, representative review history | Rows carry authority a machine cannot invent | Keep; improve executable reach and hermetic closure. |
| Backend header lanes | Phase 0 closure precedent and device-lane preprocessing probes | Host SDK leakage and zero-row Metal lane | Adopt hermetic attributed closure before relying on the lane. |
| Runtime-extent target manifest | Manifest code, tripwire, post-review changes | Five synchronized feature changes, no repair-only drift | Keep. |
| `.config/ci-test-targets.toml` | chelis#1824 and current history | 53 to 61 post-review growth | Do not decide here; require an admission and omission-detection rule. |
| Rejected-cells corpus | File contract, unsupported structure, collect-all and mutation probe | Byte drift versus stable identity, cross-lane skew | Structured identity blocking; wording snapshot reviewed. |
| Dropout gate rows | Gate implementation and entry-path findings | Real behavior changes and policy-fork detection | Keep, with per-target and per-entry claims. |
| Rejection-authority liveness | Design §C2.1/§C7.5, workflow, validator | Useful audits, bystander failures, missing scheduled half | Keep membership; deliver scheduled standing check before narrowing PR liveness. |
| Narrow-float messages | Dtype matrix, compiler copies, chelis#1871 | Triplicated prose disagrees with normative source | Share identity and point to the matrix instead of restating it. |

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
entries. At `23729c638` it contained 61.

| Merge | Added targets |
|---|---:|
| chelis#1894 | 3 |
| chelis#1918 | 3 |
| chelis#1926 | 2 |

The additions are legitimate, but each came from the pull request currently
needing the test. No rule derives omissions or states which future test target
must be selected. The measurement strengthens chelis#1824's decision request;
it does not determine which of that issue's execution policies is best.

## Current executable reach

Checked on `main` at `23729c638`:

| Artifact or test | Reach |
|---|---|
| `expand_call_site_inventory` | Absent from `.config/ci-test-targets.toml`; reached by daily/manual broad jobs |
| `expand_insert_source_literal_inventory` | Present in `.config/ci-test-targets.toml` and reached by daily/manual broad jobs |
| `issue_687_rejected_cells_corpus` | Present in `.config/ci-test-targets.toml` and reached by daily/manual broad jobs |
| `phase3_gate_contract` | Present in `.config/ci-test-targets.toml` and reached by daily/manual broad jobs |
| `capacity_census_wire` | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 oracle |
| `capacity_census_bindings` | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 oracle |
| §C7.5 `loud-unsupported-nightly.yml` | File absent |
| current rejection-issue validator | Runs in `ci.yml` against the standing manifest |

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

## Current-main extended-run evidence

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
The current run failed ten distinct rows:

| Failure class | Distinct rows | Current observation |
|---|---:|---|
| Chained expand | 4 | Host evaluation or C agreement panics because a synthetic dimension resolves to conflicting actuals. |
| chelis#345 gradient | 1 | Evaluation rejects construction of the backward DAG at same-shape/same-dtype `relu` verification. |
| Module-export fixture precedence | 3 | One proof fixture and two opaque-type fixtures encounter an unexported import before the result their assertions expect. |
| `annotated_concat_softmax.ch` corpus validity | 2 | Strict Deep validation and `--desugar` reject the executable example added by chelis#1918. |

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
  census and parity test.

The first two classes have explicit owning claims about exact observable
rendering. They are not evidence that all prose should become identity-only.

The chelis#1918 unsupported copies do fall within chelis#1870's design question
because the numbered unsupported contract identifies structured fields. The
implementation review must decide whether the executable example intentionally
adds a stronger exact-text contract. Until that decision, the sites belong in
the migration inventory and must not be silently weakened.

## Evidence still missing

The review did not establish:

- local guard-firing frequency where pull-request bodies record only the final
  green result;
- complete review-thread behavior for every historical digest move;
- whether atom anchors continue to grow without the current oracle-file edit;
- primary backend-header exposure under a hermetic SDK closure;
- the runtime and cost of a scheduled standing-liveness canary; or
- the best target-selection policy under chelis#1824.

Each implementation slice should close only the unknowns it needs. None of
these gaps justifies a broad “remove all ratchets” or “retain all ratchets”
decision.
