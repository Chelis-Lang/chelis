# Changing tests, inventories and protected contracts

Read this guide before starting PR work and again before merging. Follow the
sections relevant to changes in tests, generated lists, error messages, build
workflows or protected project rules. The owning specification still decides
correct behavior; this guide explains how to make the change reviewable.

Run the focused checks for your change and the normal pre-push gate:

```sh
python3 scripts/gate.py --fast
```

CI selects additional checks automatically. A passing fast gate does not certify
all compiler behavior or replace a named acceptance command required by the
owning design.

## Candidate lifecycle

Immediately before the first push, fetch the actual target and rebase onto it
unless the branch is already based there. Run focused checks and
`python3 scripts/gate.py --fast`, then publish the initial review candidate.
During a review round, consolidate its findings into a local commit and hand
that exact unpushed head to the standing reviewer. Do not push repair commits
until the reviewer closes the round. Amend or replace the local commit as needed
and repeat verification there. Once the round is satisfied, push the exact
verified head once and let CI validate it. A confirmed in-scope P0 or P1 then
requires a fresh round, subject to the repository's round cap.

After the pull request exists, do not merge or rebase the target branch merely
because it advanced. If a real conflict, unsafe prospective merge, or identified
semantic overlap requires a base update, put exactly one line for the new local
head in the PR body before pushing:

```text
Candidate-base-update: <new-head-sha> <specific conflict or semantic reason>
```

An approved force-pushed rewrite that does not move onto a newer base uses:

```text
Candidate-history-rewrite: <new-head-sha> <specific approved reason>
```

The candidate preflight rejects a missing, duplicate, empty, or stale-head
declaration before expensive CI starts. A PR-body edit does not restart compiler
CI. If the declaration was omitted, add it and rerun the failed workflow on the
same head rather than creating another candidate change.
Workflow-native path selection independently forces the CI-contract suite when
the candidate detector is itself under change. A failed or unavailable preflight
suppresses expensive work but fails the required Docs and Hull contexts.

A rebase does not by itself require a fresh round. If its hand-resolved
intersection stays within files and mechanisms the standing reviewer already
read, send only that intersection back for focused verification, including
semantic conflict resolution. Use a fresh round when the rebase introduces a
new mechanism, touches files that reviewer did not read, or materially broadens
the reviewed surface.

On a base-changing `synchronize`, the trusted rebase verifier may reuse prior
evidence when the previous head has an eligible default-branch receipt, the
target is unchanged, the base advanced strictly forward, the new head contains
that base, and the current synthetic candidate is exact. The complete
incremental validation delta runs from the prior receipt's synthetic candidate
to the current one, covering both conflict resolution and any target advance
included in that frozen candidate. `event.before..event.after` remains the
head-rewrite record for review. The prior receipt already covers the earlier PR
patch, so the cheap docs lane requires only that complete synthetic-candidate
delta to be documentation-only; the PR itself may contain code. A code-bearing
delta is mapped to exact packages and reviewed owner jobs. Package seeds expand
through reverse workspace dependencies, and the targeted lane runs
package-scoped Clippy, formatting, default-feature library/binary units and
existing doctest owners, all eligible integration targets in that package
frontier, and only the additional Python/script, SMT, backend, diagnostic or
Hull owners selected by the frontier. A Rust-policy owner with no package
frontier runs the complete `lint-and-unit` stage. The trusted verifier applies
the planner's exact Cargo target identities and package/path rules before
fan-out; ambiguous targets, paths owned only by standing or nightly evidence,
CI-policy changes, retargets, missing receipts, non-forward updates, unmapped
paths, and uncertain history run full CI. That includes a changed `ci-fast`
input because targeted rebases otherwise skip that standing job. The
fail-closed full lane overrides the ordinary docs-only skip. Every required
context still reports on the new head.
Patch changes and path overlap are reported so the standing reviewer can
inspect the resolution; they do not alone force unrelated work to rerun.
If the target advances after GitHub creates a pull-request event, CI binds the
actual checked-out synthetic candidate rather than failing against the stale
event-base SHA. The candidate must still have exactly two parents and its second
parent must be the event's exact PR head.

## Final package-expansion dispatch

After review findings are resolved and no further content change is planned,
dispatch `PR Package Expansion` with the pull request number and exact head SHA.
Do not serialize it behind the final required implementation checks: run
both at the same time, then inspect both before merging.

```sh
pr=1234
head=$(gh pr view "$pr" --json headRefOid --jq .headRefOid)
gh workflow run pr-package-expansion.yml --ref main \
  -f pr_number="$pr" \
  -f expected_head_sha="$head"
```

The workflow rejects a closed pull request, a malformed SHA, a head that no
longer equals `expected_head_sha`, or a target-branch retarget. Its planner
freezes the exact synthetic merge SHA and target branch name used by every
worker and the summary. Later movement of the same target branch does not
invalidate that frozen candidate. Inspect the `Manual Package Expansion
Summary`, then record the reviewed SHA and run link in the pull request.
Resolve failures introduced by the candidate. Record inherited failures and
missing, timed-out or otherwise incomplete coverage explicitly; a summary
without complete successful receipts is not evidence that the selected tests
passed.
A review round does not run package expansion merely because it exists.
Intermediate review candidates use required CI; expansion starts once on the
settled reviewed head. An ordinary content change after expansion, a
base-branch retarget, or a rebase the trusted verifier does not accept requires
a fresh package-expansion dispatch. A verifier-accepted rebase, including a
hand-resolved conflict, retains the successful expansion from the receipt's
previous head after the complete delta-selected checks pass and the standing
reviewer inspects the resolution. Record its SHA/run with the rebase decision
and current required-check run. Do not rebase a ready pull request merely to
refresh it after `main` advances. If GitHub can safely merge the exact reviewed
head, preserve that head and its existing evidence; inspect the prospective
merge as described in
[Worktree And Branch Discipline](../AGENTS.md#worktree-and-branch-discipline).
A base-branch retarget still requires the fresh coordinated implementation
validation described below.

`PR Contract Acknowledgements` separately owns acknowledgement enforcement.
Editing the title or description reruns that required check without cancelling
or replacing implementation results for the unchanged commit; generic edits do
not enter compiler or Hull workflows. A base-branch retarget is not
metadata-only. `PR Base Retarget Validation` holds the head pending while fresh
compiler and Hull dispatches validate the exact new synthetic merge. Wait for
that receipt and the refreshed acknowledgement/changelog checks.

### Reading the expansion report

The report differences every observed failure against the newest complete
nightly default-branch run and reports three disjoint counts: introduced,
inherited, and unrun. It is clean when nothing was introduced, so a clean
report is a real result rather than an absence of coverage, and a red one
names the rows to act on. Read the counts, not the shard timings. An
introduced row marked `absent` means the baseline never ran that test, so it
confirms nothing and you establish the verdict yourself. Inherited rows belong
to the default branch, not the candidate; do not repair them in the pull
request. An unrun count is coverage the lane did not reach, never evidence of
a pass, and it is the number to cite when a claim needs the coverage the
expansion did not deliver. A report that says it had no usable baseline has
classified nothing: it is not clean. When the message is that no run retains
every artifact, redispatch once a complete nightly exists; when it is that no
run is contained in the candidate's base, a later nightly is further away
rather than nearer, so update the candidate's base instead, under the rules
above.

The baseline is the newest complete nightly the candidate's base contains,
which is usually a few commits behind that base rather than equal to it. Over
that distance the classification errs in both directions, and only one of them
is safe. A test that went red on `main` after the baseline is reported as
introduced, so an introduced row you cannot attribute to the candidate is worth
checking against `main` before repairing it. A test that was failing at the
baseline, was fixed on `main` since, and is broken again by the candidate
reports as inherited, so an inherited row is not proof the candidate is
innocent of it. The summary prints the distance; treat a large one as a reason
to read the rows rather than the counts.

## Adding, moving or removing tests

For an existing command whose selected tests are protected, adding another test
is allowed: every previously required test and every newly selected test must
remain enabled, execute and pass. Do not rewrite the required list merely
because the selected set grew.

Deleting, renaming, ignoring or filtering out a required test still fails. Restore
its coverage, or review an intentional replacement and update its owning
contract and required identities together. Regenerating the list to omit a
missing test is not a repair. Commented-out test declarations do not count as
present.

New or directly modified default-enabled Rust integration targets enter required
PR CI automatically. Use exact `package::target` identities when inspecting the
plan; equal target names can belong to different packages. A change to a helper
or implementation file does not imply that every affected integration test ran:
broader package execution is currently informational. Run your focused behavior
checks. Feature-specific, ignored or hardware-dependent coverage needs its
explicit owner; do not hide a failing target with an exclusion.

If a directly modified target contains only ignored tests, it still fails closed
unless `.config/ci-test-targets.toml` gives that exact target a reviewed
`manual_only_target` execution mode. That mode executes the complete ignored
suite and records exact per-test results. It becomes stale if an active test is
added and cannot be combined with a target or test exclusion. When the ignored
tests need prerequisites no Linux PR worker has, a `manual_gate_target` row
instead cites every `docs/manual_gates.md` entry whose command runs that exact
target, and one of them must run all its ignored tests with no name filter; the
same rules apply, and the report lists the ignored tests, runs none, and records
the target as a manual gate not executed in PR CI.

If planning reports an unknown shared path, add a reviewed mapping to its real
packages or existing automated owner in `.config/ci-test-targets.toml`. Do not
add an unrelated mapping just to satisfy the planner. The standing test list is
a reviewed coverage decision, not a list to extend for every new test. See
[CI ownership and execution](ci_validation.md).

A completely removed shared file may retire its routing row in the same change.
The planner records `shared_path_deleted` only for a path proven present in the
base tree and absent from the candidate; the local classifier checks the same
base membership and working-tree absence. An added, modified, renamed-to, or
still-present path still needs ownership. Retirement does not classify code as
documentation or waive ordinary CI and protected-contract deletion checks.

For a new executable example under `examples/`, add its per-file parity test.
Pass its literal root path to the existing parity/check helper, directly or
through an immutable owned-path local. Borrow it only when passing it to the
helper; reference-valued local aliases are outside the accepted input grammar.
Put helper declarations at module scope: local items in an input-bearing test
are rejected. The completeness check derives membership from those inputs,
without a second filename list. The test still owes its applicable execution
and comparison checks.

Each protected test must also retain its designated comparison or harness call
and the reviewed executable/library mode. CI checks parsed calls in each body;
comments, strings, unused closures and macro examples cannot replace them.
Keep helpers at module scope and avoid shadowing their names inside a protected
test. Result-returning comparisons must use the reviewed direct check: unwrap
success (or panic directly on error), assert the expected success/failure,
expect an error for a rejection canary, or compare the operation mapping with
`assert_eq!`. Discarding the result or swallowing its error fails. Keep these
checks directly on the call; routing through a local/helper needs a reviewed
extension to the accepted syntax. The check does not prove arbitrary execution
or complete test semantics, so retain the behavioral assertions and full owning
acceptance checks. Ordinary body edits do not require checksum updates.

## Updating an inventory

First determine what each row means.

- A generated list of what exists in source should be regenerated with its
  owning tool and the resulting diff reviewed.
- A reviewed classification, required-coverage list or exclusion policy records
  a decision. A generator cannot invent that decision or remove its obligation.

Moving a typed-vocabulary consumer within its owning crate does not require a
new file-location pin. Removing the crate's last real evidence still fails;
comments and literal text cannot supply it. Adding a valid `expand` use does not
require updating an occurrence count: the behavioral and dispatch-position tests
own the retained obligations.

A newly discovered numeric export still needs the exact authority required by
[Numeric Surface Discipline](../AGENTS.md#numeric-surface-discipline). Nested
published headers are included in discovery. Changing a baseline to include an
unclassified export does not authorize it.

## Changing a protected rule or test

Protected rules and contract regions do not need checksum updates. Review
the underlying change, its consumers, and its positive and negative behavior
evidence, then supply the exact acknowledgements below. Required identities,
region boundaries, literal clauses and semantic checks remain blocking; an
acknowledgement cannot waive them. The temporary Nix workflow and the completed-contract freezes keep their
existing digests.

After committing a contract change, obtain the exact required PR-description
lines with:

```sh
.venv/bin/python scripts/phase4b_change_report.py --base origin/main --output target/phase4b-contract-changes.json
```

The report compares committed revisions. Uncommitted edits are not its input.
Read `required_acknowledgements` in the JSON and put each listed line exactly
once in the PR body, outside a code fence. For example, a changed numeric rule
can require both:

```text
Frozen-contract-change: spec/05-risc-primitives.md
Frozen-contract-change: atom:05-OP-33
```

A changed protected region uses its exact label as a JSON string, for example
`Frozen-contract-change: region:"agent numeric surface discipline"`, plus its
changed-file line. Use only the lines your report requires: missing, duplicate,
stale and unknown acknowledgements fail. Registry edits can require an owning
atom's line even when the chapter itself was not edited. Removed declarations
remain review obligations.

The lightweight enforcing check accepts the saved PR body:

```sh
.venv/bin/python scripts/phase4b_change_report.py --base origin/main --output target/phase4b-contract-changes.json --require-acknowledgement --acknowledgements-file target/pr-body.md
```

The full frozen-contract oracle (`scripts/dtype_phase4b_oracle.py`) remains an
independent compiler-contract check; an
acknowledgement cannot replace it. CI supplies its own validated PR comparison.
Do not use `--pr-head` on an ordinary local branch.

For protected Rust tests, generate the required review lines after committing:

```sh
.venv/bin/python scripts/phase3_test_change_report.py --base origin/main --output target/phase3-test-changes.json
```

Begin the PR description with each `required_acknowledgements` line exactly
once, in one contiguous block. Put a blank line between that block and your
problem description. Lines later in the body do not count. For example:
`Protected-test-change: crates/chelis-cli/tests/parity.rs::parity_corpus_is_complete`.
CI requires the exact changed set, including removed requirements, and rejects
missing, duplicate, stale, unknown or malformed lines. Editing the PR description
reruns `PR Contract Acknowledgements`; keep the lines in the final description
and wait for that check before merging. To check a saved body,
add `--require-acknowledgement --acknowledgements-file target/pr-body.md` to the
command above. The dedicated acknowledgement check publishes its enforcement
result, while Docs retains the independent non-enforcing comparison report.

Review those changes and retain the required body/result checks, comparator
calls, receipt checks and negative controls. An acknowledgement
cannot waive a missing definition or another guard's rejection. A changed
mutation witness still owes review of its implementation, command and expected
rejection; producing the same error alone does not prove the witness stayed
strong enough.

## Changing an error message or its issue reference

Preserve the rejection's subject, compiler stage, diagnostic kind, authority,
source location and supported alternative where applicable. Existing exact-text
checks and their negative controls remain until the owning contract replaces
them with complete structured checks. Some public error contracts deliberately
require exact bytes.

For a message already using a shared reviewed expectation, update that one
expectation and regenerate its artifact with the owning tool, then run its
consumer tests. The current softmax wording generator is
[`generate_reviewed_unsupported_wording_snapshot.py`](../scripts/generate_reviewed_unsupported_wording_snapshot.py).
It generates a reviewed expectation, not text inferred from the production
emitter; a changed emitter does not automatically make its new wording correct.
Do not add another parser for the rendered sentence.

After changing a production `unimplemented_rejection!` citation, regenerate:

```sh
.venv/bin/python scripts/generate_rejection_registries.py --write
```

Review that the issue is relevant to the missing implementation. PR validation
is split across existing owners: `script-unit` checks privacy and source-usage
construction, while the always-running Docs frozen-contract oracle freshly derives
compiler closure and generated-registry agreement. Neither queries GitHub. The
scheduled canary checks every source-derived identity and rejects closed,
missing, or pull-request references; run it manually as well before release or
relevant review claims. Do not re-cite unrelated code merely to quiet that
report.

## Changing the guarded Nix workflow

The temporary event restrictions and byte freeze still apply. A recipe change
needs an explicit reviewed freeze update. Removing the freeze belongs in the
reviewed replacement-policy change, after its event/recipe controls and complete
native flake and Devenv smoke checks pass on both supported systems. Evaluation
alone does not meet that condition.

[The assessment](../spec/design/guard_artifact_proposal_assessment.md) records
these selected contracts and the work still required before further protections
can be retired.
