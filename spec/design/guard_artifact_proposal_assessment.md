# Guard-Artifact Proposal Assessment

**Status:** ACTIVE implementation guidance for chelis#1868, chelis#1869,
chelis#1870, and chelis#1824. This document does not amend the agent contract,
a numbered specification, or an executable oracle. It records which assessed
slices have landed and the selected delivery contract for the remaining
pull-request execution work.

**Related decisions:** chelis#1824 decides where pull-request validation runs.
Chelis#1882 reports that the wire and bindings capacity censuses run in no
automated job, but that premise is stale on the assessed head: the daily and
manual heavy-E2E dtype job owns both censuses.

**Evidence snapshot:** the original review was written on 2026-09-12 from
`main` at `0e858f184` and later evidence through `72a1fb497`. This repository
copy was first refreshed on 2026-09-13 against `23729c638`, then refreshed
again against `main` at `8d1293f31` after the implementation sequence through
chelis#2008. The CI implementation refresh uses `main` at `b38efcfbe`
for the current integration inventory. The companion
[`guard_artifact_proposal_evidence.md`](guard_artifact_proposal_evidence.md)
records coverage, measurements, and known limits.

The delivery status below is refreshed through chelis#2053 at merged `main`
`929b2c483`. Earlier inventory sizes and extended-run failures remain explicitly
dated evidence, not measurements of that later tree.

The refresh also corrects an execution-reach error in the local report and the
first repository draft. The heavy-E2E workspace shards exclude
`capacity_census_wire` and `capacity_census_bindings` because the dedicated
dtype Phase 0-3 oracle executes them. They remain absent from pull-request
`ci-fast`, but they are not local-only.

## Decision summary

| Proposal | Recommendation | Required correction or condition |
|---|---|---|
| chelis#1868, frozen digests | Selected replacements implemented | chelis#1959 landed selection-superset semantics; chelis#1969 split the Phase 0 freeze and chelis#2017 completed its mutation-rejection binding. Corpus/body checks and mandatory review acknowledgements replace definition hashes through chelis#2040, chelis#2050, chelis#2051 and chelis#2053. Atom/region hashes were retired in chelis#2045. The selected Nix decision is retention with an explicit retirement condition, delivered in chelis#2033. |
| chelis#1869, inventories | Core assessed slices implemented | chelis#1960 kept crate-granular positive vocabulary evidence, chelis#1963 closed backend-header discovery, and chelis#1968 retired both redundant expand inventories; chelis#1998 removed their stale CI row. Reviewed semantic classifications remain non-regenerable. |
| chelis#1870, diagnostic pins | Assessed trial implemented; broader migration remains | chelis#1961 derives issue membership, chelis#1971 supplies the scheduled standing-liveness canary, chelis#1972 shares narrow-float capability identity, and chelis#1973 landed the structured unsupported-identity trial while retaining rendered compatibility. |
| chelis#1824, pull-request execution reach | Required lane implemented; expansion selected as an explicit final-candidate dispatch | chelis#2019 delivered package-qualified planning/execution and standing ownership of both replacement expand controls. Automatic PR/main expansion did not fit its cost envelope. The selected follow-up keeps nightly coverage, moves expansion to a stale-head-rejecting manual workflow, and separates acknowledgement enforcement from compiler CI. Execution batching and receipt deduplication remain a separate delivery slice. Local `--fast` remains unchanged. |
| chelis#1882, scheduled census reach | Closed with executed evidence | The daily/manual dtype Phase 0-3 job executes both censuses. The issue's demand for a citable automated verdict is implemented; per-pull-request reach belongs to chelis#1824. A recorded failing census proves reach, not correctness. |

The governing distinction is:

> A guard list carries a review disposition per row or is regenerated from the
> tree. A hand-typed copy of what the tree contains is not a guard.

The frozen-digest corollary is:

> A digest is useful when it binds a reviewed claim that cannot be reconstructed
> from the current tree. A digest over ordinary tree growth is a review trigger,
> not semantic evidence, and may be replaced only by a derived trigger that
> preserves the same obligations.

## chelis#1824: change-owned and changed-package pull-request execution

The selected policy is neither a longer hand-maintained target list nor a full
workspace run on every pull request. It adds a precise change-owned target
guarantee and a broader cost-bounded changed-package trial beside the current
standing lane.

### Cadence and ownership

| Surface | Result after implementation |
|---|---|
| Developer `gate.py --fast` | Unchanged: tier-0 regeneration, formatting, lint, changed-crate Clippy, and 13 fixed integration tripwire identities. It remains the pre-push gate and does not become a broad integration run. |
| Hosted `gate.py ci-fast` | Preserve the standing baseline of every default-feature library/binary unit target plus the reviewed integration manifest under its existing 20-minute job limit. The base manifest has 73 package/target identities at `b38efcfbe`; this implementation adds `chelis-types::expand_insert_dispatch_family` and `chelis-types::issue_1294_standard_lowerings`, bringing the candidate standing set to 75. |
| New hosted change-owned lane | On every non-doc pull request and main push, run every default-enabled integration target added or directly modified by the change. An exclusion is valid only when it names an exact alternative owner and reason. |
| Explicit hosted package expansion | After reviews and repairs fix the intended content, agents dispatch separate workers alongside the final required implementation checks. They run every other default-enabled integration target in each directly changed package or package selected by a reviewed shared-path rule, except exact reviewed target/test exclusions. The workflow accepts a PR number and expected head SHA and rejects stale dispatches. Do not create a new synthetic candidate merely to refresh a ready branch after `main` advances; preserve the exact reviewed head when its prospective merge is safe. |
| Linux nightly | Unchanged full backstop: all non-ignored default-feature workspace tests across four shards, with the two capacity censuses still deduplicated into their dtype owner and the existing explicitly invoked ignored/manual suites retained. |
| macOS and feature/nightly owners | Unchanged. The new Linux lanes make no cross-platform, non-default-feature, hardware, ignored-test, or phase-acceptance claim. |

The standing lane remains useful even when no package maps from a documentation
or workflow-only change. The change-owned lane closes the structural omission
where a pull request adds or directly changes an integration target root but no
reviewed row names it. Package expansion may catch same-package helper and
implementation effects without pretending to compute a reverse-dependency or
semantic impact closure. Nightly remains the completeness net for packages the
pull request did not directly change and for exclusions too expensive or
prerequisite-bound for the regular lane.

The two standing additions resolve the concrete post-chelis#1968 case
recorded on chelis#1824. The retired spelling inventories no longer belong in
CI, but their small behavioral replacements did not inherit a regular
pull-request owner. `chelis-types::expand_insert_dispatch_family` protects the
two AST dispatch positions, while
`chelis-types::issue_1294_standard_lowerings` protects the behavioral
expand/insert distinction. The latter also runs in the broader dtype builtin
atom closure, but that is not ordinary per-pull-request selection. On
`da11a6fa3`, the two targets ran 12 tests in 0.060 seconds of measured nextest
execution. Their standing admission is an explicit ownership decision, not a
request to restore copied counts or a precedent that every same-package test
must enter the standing layer.

### Planner contract

The planner must fail closed and emit one final disposition for every changed
path:

- an exact added/directly-modified integration target;
- one or more selected workspace packages;
- a named standing CI or nightly owner for a reviewed shared/non-package path;
  or
- the existing docs-only disposition.

Unknown paths, malformed configuration, stale package names, duplicate rules,
ambiguous mappings, and paths with no disposition are planning failures. The
first implementation should use a versioned configuration with four explicit
row kinds:

- `standing_target`: the base 73 package-qualified reviewed identities plus
  the two explicit replacement-expand additions above;
- `target_exclusion`: one exact package/target excluded from the change-owned
  or package-expansion lane;
- `test_exclusion`: one exact package/target/test excluded from an otherwise
  selected target; and
- `path_rule`: a shared path mapped either to exact packages or to a named
  existing owner.

Each exclusion must name its workflow, job, cadence, reason, and tracking issue.
The three whole-target and six exact-test heavy exclusions currently expressed
by the default nextest profile need such rows before package expansion runs
with `--ignore-default-filter`. A newly added or modified excluded target still
must resolve to that exact alternative owner. Cost alone is not an admission
rule and no slow test is promoted to nightly merely because it crosses a
timing threshold.

For pull requests, the checked-out synthetic merge commit is the execution
candidate. The planner must require exactly two parents, use `HEAD^1` as the
base, require `HEAD^2` to equal the event pull-request head, and derive the
NUL-delimited rename-aware change set with
`git diff --name-status -z --find-renames BASE CANDIDATE`. A main push uses the
event's before/after commits and the same path-classification logic.

Base and candidate Cargo metadata define exact `package::target` identities and
their `src_path`. A candidate identity absent from the base is added; an
identity whose candidate `src_path` is in the changed-path set is directly
modified. A renamed target is a deletion plus an addition. Changes to shared
test helpers, package implementation, examples, scripts, specifications, or
workspace files are not mislabelled as direct target changes; they enter
package expansion or an explicit shared-path disposition.

A target whose `required-features` are all enabled by that package's
default-feature closure is eligible; a target requiring a non-default feature
is not. Package qualification is mandatory: `main` at `b38efcfbe` has 779
default-eligible integration targets in a 34-package workspace; 24 packages
currently own at least one. Thirty-six target names, covering 78 identities,
are shared by more than one package, so execution must preserve package
qualification. The standing runner groups
globally unique names with units and runs shared names under exact package
selectors, merging their listing and execution receipts.

### Execution and receipts

The planner writes one machine-readable plan containing the candidate and base
SHAs, a normalized ownership-configuration digest, raw changed-path records,
path dispositions, selected packages, eligible targets, the exact change-owned
subset, the standing-coverage reuse subset, the disjoint package-expansion
subset, target/test exclusions, exclusion owners, and a plan digest. Planning
is required: malformed or incomplete classification blocks both execution
surfaces.

Required change-owned execution and manually dispatched package expansion must
not share worker budgets or aggregate status. Exact `package::target` identities in
each subset are independently assigned to four deterministic shards by
`sha256(package + "::" + target) mod 4`. Runners execute package-scoped Cargo
selectors so equal target names in different packages cannot create a cross
product or ambiguity.

Each required change-owned shard has a 20-minute hard timeout and uploads its
command, selected and executed test lists, timings, JUnit, and plan digest. Its
required report fails on a missing shard, digest mismatch, duplicate execution,
uncovered selected target, executed exclusion, or non-success result. The
stable `Integration Tests (Linux)` context eventually depends on the standing
`ci-fast` worker and this change-owned report only.

A target present in both the standing and change-owned sets is not executed
twice. The standing worker writes a digested coverage record binding candidate,
normalized configuration, exact execution mode, every standing target and
every selected/executed test. The required report accepts the overlap only when
that record is complete and exact; stale candidates, changed configuration,
missing results, failures and tampering leave the change-owned obligation
unsatisfied. A plan-proven empty shard writes explicit zero-selection evidence
before dependency installation or cache restoration.

Package expansion runs only through a separate four-shard manual workflow with
its own 20-minute hard timeout and 15-minute soft telemetry budget. Agents
dispatch it alongside the final required implementation checks once reviews and
repairs have fixed the intended content, then record the exact reviewed SHA and
run link. Its summary records missing shards, timeouts, test failures,
exclusions, timings, and receipts but does not replace a branch-protection
context. A content change or hand-resolved conflict invalidates the run. A
base-changing rebase or base-branch retarget does too because it changes the
synthetic candidate. Agents do not rebase a ready pull request merely because
`main` advanced; they preserve the exact reviewed head when its prospective
merge is safe. Introduced failures are repaired; inherited failures and
incomplete coverage are named explicitly. A package-expansion timeout or
failure cannot cancel, starve, or change the verdict of required change-owned
execution.

The expansion executor has a separate 16-minute deadline, measured from its
start and shared by the product build, target listings and test commands. On
expiry it terminates the active command's process group, starts no later
command, and writes an unsuccessful receipt with the completed evidence and
the full original target selection. Unfinished targets remain uncovered; a
deadline is not an exclusion or evidence of passing tests. The existing
15-minute soft budget remains telemetry. The gap before the 20-minute job
limit allows receipt upload after normal executor budget exhaustion; setup
delays, external cancellation and runner loss can still leave a missing shard.
The required change-owned executor does not acquire this shorter deadline.
Positive and negative subprocess/receipt tests must exercise expiry during
build, listing and execution, preservation of earlier results, child-process
termination, and successful completion without changing required execution.
Within a shard, ordinary expansion targets are batched by exact package.
Manual-only targets and targets with exact exclusions remain singleton groups.
The listing and JUnit parser maps the package command back to every exact target
and test; missing per-test output leaves the affected target incomplete.

### Delivery sequence

1. The planner, versioned configuration, required change-owned report,
   separate package-expansion workers, telemetry, and the two exact
   replacement-expand standing rows landed in chelis#2019.
2. Exact-head trials established that automatic package expansion does not fit
   the regular PR/main cost envelope.
3. Land the schedule and acknowledgement slice: retain nightly coverage, remove
   automatic package expansion from PR/main CI, add the explicit PR-number plus
   expected-head dispatch, make final-candidate dispatch an agent requirement,
   and move PR-body acknowledgement enforcement into its own required check.
   Description edits rerun only acknowledgement/changelog policy and do not
   enter compiler or Hull workflows. Base retargets create a separate required
   head receipt that remains pending until trusted-base coordination dispatches
   exact-head/exact-base CI and Hull against the new synthetic merge.
4. Land execution optimization separately. The implementation candidate skips
   setup for plan-proven empty groups, reuses only complete
   exact-candidate/configuration coverage receipts, and batches expansion by
   package while preserving exact target and per-test evidence, honest
   timeouts, and partial results.
5. Collect comparable small, multi-package, script-only, and documentation PR
   measurements, including required latency, runner minutes, dollars per run,
   final expansion and post-merge spending. Remove `PROVISIONAL pending
   chelis#1824` comments and close the issue only after the selected contract
   has hosted evidence on merged `main`.

## chelis#1868: frozen digests and freeze constants

The proposal covers eight artifact groups. They do not have one answer.

### Phase 0 runtime-representation freeze: implemented with retained mutations

Chelis#1969 implemented this split. `foundation_rows` and the mutation
projection remain inside the frozen object. Each mutation row binds a witness
identity, exact implementation digest, expected failure, and command. The
executed expected-failure contract alone cannot detect a witness rewritten to
fail trivially for the expected reason.

Release reproducers, hardware registrations, registered-source counts, and
other ordinary oracle configuration are code-derived outside the digest.
Schema closure rejects extra stored authority, and regeneration rejects a
retired foundation identity that reappears without a reviewed freeze move.

This is not theoretical. chelis#1828 changed the implementation digest of 16
of 44 existing mutation witnesses without adding or removing a witness. Every
mutation still ran and passed. Only the implementation binding could reveal
that what the witnesses checked had changed.

### Phase 1 and Phase 2 selection manifests: superset semantics implemented

Reject chelis#1868's original regenerate-on-failure replacement. It would
remove the blocking check on deletion of a frozen test.

Chelis#1959 implemented chelis#1869's superset form:

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

### Phase 3 required-test definition digests: replaced with review and body checks

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

The corpus-membership slice derives the filenames from parsed Rust test inputs
and compares them with the executable `.ch` files directly under `examples/`.
It retains the existing per-example bodies and their executable, library-only,
or rejection behavior. The accepted input forms are direct top-level
calls to the parity/check helpers. Input provenance uses a closed owned-path
grammar: `examples_root().join("file.ch")`, parentheses around that expression,
or a move from a previously admitted immutable owned-path binding. Shared
borrows are allowed only at the helper argument; reference-valued bindings
never establish owned-path provenance. Every local item in an input-bearing
test is rejected, irrespective of its name or item kind: helper declarations
belong at module scope. This avoids interpreting Rust aliasing or maintaining
a partial list of value-item shadowing forms. Pattern bindings still invalidate
old identities, and conditional syntax cannot supply membership. Helper
literals, comments and strings are not input evidence. The existing ignore ledger still
owns the one declared manual prerequisite. This is input-membership evidence,
not proof of comparator execution. Tests exercise discovery, added/uncovered
examples, missing inputs, and removed or hidden declarations before the copied
filename array is removed. Its fixed completeness-test body remains frozen;
the other definition freezes and the retirement obligations above stay active.

The initial per-body trial added a Rust AST check beside the definition hashes. The
current Python oracle exports each required identity and its reviewed call role
as transient JSON; there is no second stored identity inventory. Each body must
name its designated harness, comparison or canary entrypoint, and evaluator
canaries must also name their receipt entrypoint. Ordinary evaluator agreement
rows require both agreement and expected-value checks. Executable parity rows
must pass literal `true` to the harness, while declared library rows pass
literal `false`. The Metal rejection uses the shared exact comparator to check
that no artifact was emitted, retaining its status and diagnostic assertions.

The audit visits parsed eager expressions and the condition operands of the
standard assertion macros. Comments, strings, unknown macro payloads, local
item definitions, closures, async blocks and const bodies cannot supply calls.
Local items, conditional configuration attributes and pattern bindings that
shadow a required owner are rejected. Ordinary blocks, branches and loops are
visited, so this is a structural call-adoption check, not proof that arbitrary
control flow reaches the call or that its arguments and result handling are
correct. The existing receipt and shared-helper behavior checks retain those
separate roles. The new target runs in standing CI and the complete oracle.

The retirement slice replaces the 51 definition hashes with the derived review
cue, corpus derivation and per-body audit. It adds a closed direct-result rule
where the designated entrypoint returns a comparison result: completeness and
rejected-cell comparisons must unwrap success or use a directly panicking error
closure; comparator self-tests must assert the reviewed `is_ok()`/`is_err()`
polarity; compiled-observation and width canaries must expect an error; the IR
operation mapping must be an operand of `assert_eq!`. Parentheses are allowed.
Discarded results, non-panicking fallback closures and boolean inspection
outside the required assertion cannot satisfy these roles. Passing a result
through another local or helper requires a separately reviewed contract change;
this audit does not infer data flow or expand arbitrary macros.

Before deleting the hashes, execute the existing named mutations through the
Rust audit: every empty or removed body, each forged evaluator receipt, disabled
executable parity, and the discarded completeness result. The source/ignore,
per-file comparator, runtime receipt and shared-helper behavioral controls stay
in force. Positive controls must admit ordinary unrelated body edits and test
additions without a checksum update. The required Docs acknowledgement remains
the durable review trigger for every protected definition change. This replaces
specific mutation detection and review ownership; it does not prove arbitrary
control-flow reach, operand correctness or the complete semantics of a test.
Jeff's complete Phase 0 mutation binding, closed Phase A freeze and temporary
Nix freeze are outside this retirement.

The first trial runs `scripts/phase3_test_change_report.py` in the required
Docs job and publishes its committed merge-base comparison as an artifact.
It reads both literal required-test inventories, so removing a requirement
cannot hide the old identity from the report. Changed definitions and
membership are named with source locations and the guard doctrine. Missing
comparison evidence or candidate required definitions fails closed. This report supplies the review cue; the body audit, comparator, receipt,
example-derivation and mutation controls retain their separate obligations.

The durable review cue requires each changed required-test identity to appear
exactly once in a contiguous opening block of the PR description as
`Protected-test-change: <repository-relative Rust path>::<test name>`.
The report derives the expected lines from the union of both inventories,
including removed requirements, and publishes them with the comparison.
The dedicated required `PR Contract Acknowledgements` job enforces those lines
against the validated PR merge. A PR-description edit reruns that job so
removing a line invalidates the old result, but it neither cancels nor
recreates compiler validation for the unchanged commit. Generic edits do not
enter compiler or Hull workflows. A base retarget is an implementation event:
a dedicated required head receipt remains pending until trusted-base
coordination dispatches fresh exact-head/exact-base CI and Hull runs against the
new synthetic merge. The enforcement rejects
missing, duplicate, stale, unknown or malformed acknowledgements in that
opening block. The block precedes all prose and is separated from it by a
blank line; initial blank lines are harmless. Later lines are examples or
prose and cannot acknowledge a change. This closed prologue grammar has no
preceding Markdown context to interpret, so a quoted, fenced, inline-code or
collapsed example cannot establish acknowledgement. Missing required
definitions still fail even when acknowledged.
Push/local report generation remains available without a PR body. This makes
the review cue explicit and durable; comparator, receipt and mutation
controls remain after the per-definition hashes are retired. An acknowledgement records review responsibility, not evidence
that the author executed or understood a test.

Declaration discovery and body boundaries share a Rust comment/literal scan:
commenting out a protected test makes it missing, while declaration-like text
inside comments or strings cannot supply or duplicate it. Original source
offsets and definition bytes remain the change-report input. This scan does not
evaluate Rust configuration or prove comparator execution. PR reports validate
the exact event head against the synthetic merge's second parent and compare
against its first parent, so regenerating that merge after main advances does
not attribute main-only changes to the pull request.
Both contract-report steps select the push base by event name. A pull request's
`synchronize` payload also carries `before`; that previous PR head must not
select push mode or compete with the validated PR comparison.

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

The first Phase 4B trial runs `scripts/phase4b_change_report.py` in the required
Docs job and publishes `phase4b-contract-changes.json`. It compares committed
snapshots at a unique merge base. On a pull request, it validates the synthetic
merge against the event head and uses the merge's first parent; a push supplies
its before commit. Invalid or missing comparison evidence fails closed.

The report discovers numbered atom definitions in chapters 04 and 05, reads
both revisions' literal region/protection declarations, and names additions,
removals, changed text, changed protection, and moved region boundaries. Atom
content includes its declared normative registry and the builtin identity rows
that name it. Registry prose shared by those rows contributes to each affected
owner. A changed contract file is also named when its edit lies outside an atom
or region. Source locations and comparison commits make each cue reviewable.
The existing freeze normalization applies to atom/region text; file cues retain
byte-level changes. Missing declared files, required atoms, registry owners, or
unambiguous region markers are errors. Historical oracle code is never executed.

The acknowledgement gate requires every changed report identity exactly once:
`Frozen-contract-change: atom:05-OP-33` names an atom, and
`Frozen-contract-change: region:"exact region label"` names a region using a JSON
string. Existing repo-relative file lines remain required for every changed
contract file, including edits outside atom/region boundaries. A registry edit
also names each affected owning atom. Protection and region-declaration edits
name their identities even when no contract file changes. Removed identities and contract-file declarations
remain obligations because the report reads both inventories.

The authoritative oracle parses one acknowledgement body and validates both the
file and identity legs. Missing, duplicate, stale, malformed, and unknown
identities fail the enforcing PR mode; a file acknowledgement cannot stand in
for an atom or region. The identity comparison uses committed snapshots at the
same unique merge base as the report. The PR step validates the event head and
the synthetic merge's two parents before choosing that base; push/local runs
remain advisory about acknowledgements. JSON evidence includes the exact
required lines, while the PR body records the author's acknowledgement.

The report and enforcing acknowledgement gate supply the atom/region digest
replacement. Required atom identities and unambiguous region boundaries remain
literal reviewed declarations; current declarations carry no text hashes. The
report reads the explicit historical digest format when comparing old commits,
without executing historical Python, and normalizes both formats to the same
protected identities and boundaries. Mixed or malformed declarations fail.

Every previously protected atom and region has a mutation control that names
its edit and rejects an omitted acknowledgement. Required-literal anchors and
their negative controls, semantic registrations, and fresh source/compiler
agreement remain independent requirements: acknowledgement does not excuse a
missing required clause, atom or region boundary. Additive prose changes that
preserve those requirements pass only after their exact file and identity
acknowledgements in enforcing PR mode. That is a review obligation, not an
automated proof that the prose is semantically correct. Other frozen artifacts
were unaffected by that atom/region retirement. The separate Phase 3
definition replacement is specified above.

### Closed-corpus and count ratchets

- Keep `FROZEN_PHASE_A_DIGEST` while Phase A remains closed. Its rare moves
  have carried measured row-by-row deltas.
- Keep the temporary Nix workflow digest only while its event-policy condition
  remains explicit. Retire it in the reviewed change that ends the temporary
  manual-dispatch/published-release-only policy, after replacement event and
  native-recipe controls pass and the complete native flake checks and Devenv
  smoke checks pass on both supported systems under the replacement policy.
  A recipe edit while the temporary policy still stands requires an explicit
  reviewed freeze move; it does not satisfy the retirement condition.
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

`AGENTS.md` records this distinction under Guard Inventories, including the
required-coverage-floor obligation. The assessed inventory changes below are
implemented. The final-candidate package-expansion workflow and its execution
optimization remain chelis#1824 work.

chelis#1927 is a current example. A capacity-census fixture became illegal
after module-export enforcement changed. The repair corrected the fixture and
its owning explanation while preserving the classifier, baseline, inventory,
comparison, and denominator. That is the right repair shape for a
judgement-bearing guard.

### Expand inventories: replacement and retirement implemented

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

Chelis#1968 completed that sequencing and removed both
`expand_call_site_inventory` and
`expand_insert_source_literal_inventory` after the checker-backed semantic
contract and its tests owned the property. Chelis#1998 then removed the stale
`ci-fast` row that still named the retired insert inventory. The assessment's
replacement recommendation is therefore implemented; neither inventory should
be restored as a copied count.

### Crate-granular positive vocabulary evidence: implemented

Chelis#1960 retained the `required` markers in
`closed_vocabulary_architecture.rs` and made their evidence crate-granular, so
an owner may move within the consumer crate without failing a file-path pin.

The `forbidden` list names only spellings already imagined. It cannot prove
that some typed path is used at all. Positive crate-level evidence directly
addresses the risk that an orthogonal implementation bypasses the tagged
carrier or typed vocabulary.

### Header census discovery closure: implemented

Chelis#1963 adopted the Phase 0 scanner's closure contract for backend header
lanes: committed SDK fixtures, complete attributed include closure, and
planted include-outside-universe controls.

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

### `.config/ci-test-targets.toml` becomes the standing layer

The file grew from 53 to 68 targets after the original assessment:

- three targets in chelis#1894;
- three targets in chelis#1918; and
- two targets in chelis#1926;
- four targets for caller/formal and gradient-result scope;
- one wildcard-absorption target;
- two pipe-fold placement/surface targets; and
- one symbolic-reshape regression target, offset by retirement of the insert
  inventory row.

Those rows are useful immediate protection, but still reactive growth without
an admission rule. The chelis#1824 decision above therefore freezes their role
as the standing baseline rather than prescribing a longer hand-maintained
list. New or changed package integration targets enter regular hosted coverage
through the derived dynamic lane; durable additions to the standing layer
remain an explicit review decision.

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

The compiler-API lowering adapter must classify the retained typed value,
not its rendered sentence: typed lowering rejections use `unsupported_feature`
under [05-UNS-6], while ordinary lowering diagnostics keep `lower_error`.
This correction does not supply missing capability-cell authority or authorize
retiring the existing exact-text guards.

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

### Standing canary and changed-row PR admission

Chelis#1961 made standing issue membership source-derived, and chelis#1971
landed `.github/workflows/loud-unsupported-nightly.yml` with daily/manual
execution, serialized reporting, duplicate recovery, and close-on-recovery.
The first manual default-branch receipt, workflow run `34765689930`, passed on
`da11a6fa388e7f52465cc28728513a331a85cac5` in 10 minutes 45 seconds. Its
status job completed successfully without opening a failure issue.

The sequencing preconditions are now satisfied:

1. the source-derived manifest cannot retain an uncited row; and
2. the scheduled standing-liveness canary has workflow-registry and
   trigger-loop controls.

The pull-request check now live-validates added or modified authority rows,
while still freshly checking complete source/manifest agreement on every
triggered run. Its exact synthetic merge's first parent supplies the base;
the second must equal the event PR head. Invalid Git evidence or either
manifest fails closed. Schema 1 fixes each row's kind and state, so a changed
valid identity is a new or renumbered issue number. The standing nightly
continues to check every source-derived row, including unchanged authorities
that acquire more citing sites. This delivers the issue-liveness split only;
the structured diagnostic migration and broader §C7.5 matrix remain separate.

The capacity-liveness adapter must read the final binding format already
owned by the Rust binding guard: the three native tagged transports and the
[05-OP-45] shape operation join the existing CompilerJson and nonnumeric rows.
No binding row may regain a citation or the retired native legacy disposition.
The Python reader checks this persisted shape, including exact contract/owner
pairs and capacity flags; the Rust verifier still owns fresh source discovery,
graph, codec and execution authority. No baseline is rewritten to repair the
adapter. Current final rows require no issue lookup. The shared issue-record
and tracker-fetch API remains available to rejection and prerequisite checks;
this repair does not deliver the pending scheduled or pre-merge matrix.

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
| retired `expand` spelling inventories | Correctly absent after chelis#1968/chelis#1998; they must not be restored |
| `chelis-types::expand_insert_dispatch_family` | Small AST-position replacement is selected by the candidate standing manifest |
| `chelis-types::issue_1294_standard_lowerings` | Behavioral replacement is selected by the candidate standing manifest, in addition to broader dtype closure |
| rejected-cells corpus | Selected by `ci-fast` and reached by the daily/manual broad jobs |
| Phase 3 gate contract | Selected by `ci-fast` and reached by the daily/manual broad jobs |
| wire and bindings capacity censuses | Absent from `ci-fast`; executed by the daily/manual heavy-E2E dtype Phase 0-3 job |
| §C7.5 standing-liveness canary | Daily/manual workflow checks every standing row on `main`; the regular PR job live-validates changed authority rows |
| newly added or changed package integration target | Required change-owned shards landed in chelis#2019 and passed merged-main acceptance; representative hosted cost trials remain under chelis#1824 |

The former contrast between the two expand inventories remains historical
evidence for execution reach. The selected insert inventory stayed
synchronized longer than the unselected call-site inventory, but both were
still copied counts and are now retired. Distribution determines whether a
defect is loud. It does not turn a copied count into semantic authority.

## Post-review merge audit

Every first-parent merge after the local assessment's final timestamp was
classified. The table lists the merges that changed an assessed guard
mechanism or supplied an implementation receipt:

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
| chelis#1959 | Implemented Phase 1/2 selection-superset semantics. |
| chelis#1960 | Kept positive vocabulary evidence and moved it to crate-granular ownership. |
| chelis#1961 | Derived the rejection-issue manifest from source. |
| chelis#1963 | Closed backend-header census discovery over its attributed universe. |
| chelis#1968 | Retired both redundant expand spelling inventories after checker-backed replacement. |
| chelis#1969 | Implemented the Phase 0 foundation/mutation freeze split and shrink-only regeneration contract. |
| chelis#1971 | Added the daily/manual standing rejection-liveness canary. |
| chelis#1972 | Unified narrow-float capability diagnostic identity. |
| chelis#1973 | Landed the structured unsupported-identity trial while preserving adapter rendering compatibility and reviewed exact-text contracts. |
| chelis#1998 | Removed the stale `ci-fast` row for the retired insert inventory. |

These merges implement the listed slices but do not close the broader
chelis#1868, chelis#1869, or chelis#1870 trackers.

## Historical extended validation at `23729c638`

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
in the dtype union. The §C7.5 standing-liveness canary was absent on that
historical assessed head; chelis#1971 has since delivered it. The
structured-identity recommendation has since landed as the bounded
chelis#1973 trial.

## Remaining delivery order

The required change-owned lane landed in chelis#2019, changed-row rejection
liveness in chelis#2025, and the assessed growing-artifact replacements through
chelis#2053. Their pull-request reviews and merged-main evidence are recorded
in the owning pull requests and the companion evidence appendix. Agents must
read [the PR-author guide](../../docs/guard_changes_for_pr_authors.md) before
starting work and again before merging, as required by `AGENTS.md`.

The remaining work has separate owners and acceptance conditions:

1. **Broader PR execution, chelis#1824.** Keep the required change-owned
   guarantee and nightly backstop. The chelis#2040 merged-main trial had three
   expansion timeouts and one failing receipt, so regular PR/main expansion is
   replaced by exact final-candidate dispatch. Complete the separate empty-plan,
   verified-overlap and package-batching slice, then collect comparable hosted
   cost receipts. A successful summary with missing or partial receipts does
   not mean its tests passed. Standing-admission decisions and provisional-row
   disposition also remain with this issue.
2. **Complete unsupported identity, chelis#1870.** Retain the exact-text locks
   until the typed payload carries every obligation listed above through the
   machine channel. The capability tables and their derived consumers belong
   to chelis#729; its chelis#1296 composite acceptance is required before
   table construction. The completed chelis#1294 atom closure alone does not
   satisfy that entry condition. Preserve the diagnostic-channel convention
   and producer/transport split owned by chelis#883 instead of adding a prose
   parser or parallel payload. The wider scheduled/change-gated authority
   matrix remains chelis#990; chelis#2052 only repairs its current binding
   inventory reader.
3. **A future Nix policy change.** The decision in chelis#1868 is to keep this
   freeze while the temporary event policy stands. Changing that policy must
   satisfy the replacement controls and both native-platform checks described
   above before retiring the digest. That conditional future change is not
   part of the completed growing-artifact replacements.

These are coordinated decisions about derivation, reviewed authority and
diagnostic identity. Completion of the assessed chelis#1868/chelis#1869
replacements does not certify a compiler phase, all package tests, or the
broader chelis#1870 migration.

## Non-claims and open evidence

This assessment does not establish:

- how often local `--fast` tripwires fire when no pull-request text records it;
- whether reviewers re-read every changed atom when a digest moves;
- whether required-literal anchors would continue to grow without the current
  digest round trip;
- steady-state standing-liveness cost beyond the first successful
  10-minute-45-second default-branch receipt;
- the optimized manual package-expansion cost distribution across
  representative exact-head pull requests;
- whether shared-path rules need additional package mappings after the
  informational trial; or
- the evidence threshold for adding a target permanently to the standing
  layer instead of leaving it dynamically selected.

Those unknowns are reasons for the informational trial and its receipts, not
reasons to silently remove coverage or resume indefinite target-list growth.
