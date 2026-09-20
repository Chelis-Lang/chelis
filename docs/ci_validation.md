# CI validation cadence

For the actions to take when preparing a PR, read
[Changing tests, inventories and protected contracts](guard_changes_for_pr_authors.md).

Ordinary PRs and main pushes use Linux. Passing required PR checks is **not a phase acceptance result** for a full or feature-specific oracle that runs nightly.

| Owner | Cadence | Coverage |
|---|---|---|
| `ci.yml` / `conformance.yml` candidate preflight | Every PR implementation event; CI-contract tests only when their path classifier fires | Rejects undeclared base merges, base-changing rebases and other history rewrites; runs bootstrap-light workflow/routing tests before expensive build fan-out |
| `ci.yml` `ci-fast` | Ordinary PR and main push, with the existing docs-only skip; accepted targeted rebases use the interaction frontier instead | Every default-feature library/binary unit target and the reviewed `standing_target` identities in `.config/ci-test-targets.toml`; 20-minute limit |
| `ci.yml` change-owned shards and report | Ordinary PR and trusted exact-candidate workflow dispatch, with the existing docs-only skip; accepted targeted rebases run only when their package frontier selects integration coverage | Every integration target added or directly modified by the candidate, with its Cargo-declared required features activated, or its exact reviewed alternative owner; targeted rebases require every eligible target in affected packages and their reverse workspace dependents; four deterministic shards with a 20-minute limit each |
| `pr-package-expansion.yml` | Manual dispatch after review repairs, parallel with final required checks, using an open PR number and exact expected head SHA | Other integration targets in directly selected packages, with their Cargo-declared required features activated and exact reviewed target/test rows excluded; four informational shards with a 20-minute hard limit and a separate summary that classifies every observed failure as introduced or inherited against the nightly default-branch baseline and counts unrun coverage on its own |
| `pr-contract-acknowledgements.yml` | PR open, synchronize, reopen, title/body edit or base retarget | Dedicated required validation of persistent candidate-lifecycle, protected-test and frozen-contract acknowledgement lines; no compiler build |
| `pr-base-retarget.yml` | PR open/synchronize/reopen plus base-retarget coordination | Required head receipt. Ordinary candidates defer to the normal required implementation contexts. A base retarget holds the head pending while trusted-base coordination dispatches exact-head/exact-base CI and Hull runs against the new synthetic merge |
| `pr-candidate-receipt.yml` | Completion of any workflow that can finish the required PR check set | Default-branch-owned receipt binding the exact PR head, synthetic candidate, patch identity, required check runs and their workflow/job provenance; eligible receipts can authorize the guarded targeted-rebase lane |
| `ci.yml` retained workers | Ordinary PR and main push; accepted targeted rebases run only the owners selected by their interaction frontier | Rust policy and doctests, Python/script units selected by `ci_script_tests.py pr`, focused SMT plus its existing Deep-obligation integration target, Linux glibc compatibility, Docs, backend sanitizer units and explicit backend doctests; change-triggered diagnostic mutation and the offline rejection-authority boundary |
| `conformance.yml` | PR and main push | Existing frozen Hull conformance gate |
| `heavy-e2e.yml` | Daily 03:17 UTC and manual dispatch | Full non-ignored default workspace across four workspace shards, an exact feature-enabled owner for the known-red native Random observer debt, plus the dtype owner, script integrations, exhaustive generalization feature partitions, dtype Phases 0–3, faithful observation Phase 2, ownership Phase 2 and launch, runtime representation Phase 0, frontend/domain support, and full backend sanitizer integration coverage |
| `macos-nightly.yml` | Daily 04:17 UTC and manual dispatch | Both Mac workspace partitions, both Clippy configurations, architecture/ABI/Metal smokes, and Darwin SMT |
| `build-cvc5.yml` Darwin producer | Daily 01:17 UTC and manual dispatch | Missing Darwin prebuilt assets; relevant main pushes may produce Linux assets only |
| `smt-full-prove.yml` | Existing nightly/manual cadence | Full SMT proof validation |
| `release.yml` and release-triggered Nix checks | Release/manual cadence | Existing shipping-artifact validation, including Mac |

The Linux workspace worker passes `--ignore-default-filter` and deliberately includes the PR selection. Only `chelis-compiler-api::capacity_census_wire` and `chelis-python::capacity_census_bindings` are excluded: the dtype worker executes both. The generalization worker uses the same census exclusions; census authority is checked on the default-feature configuration. Executable listing set-math proves that workspace plus dtype still covers every non-ignored test in the unfiltered corpus, with none in neither selection and no census test in both. The two selections do overlap elsewhere: the dtype oracle also owns several non-census binaries.

The regular candidate planner runs for pull requests and trusted exact-candidate workflow dispatches. It requires the checked-out synthetic merge commit to have exactly two parents and requires the event head to be `HEAD^2`. Its rename-aware NUL-delimited diff assigns every changed path one final disposition. Package roots map through Cargo metadata, reviewed shared paths map through `path_rule` rows, and unknown, stale, duplicate, or ambiguous mappings fail planning. Added targets and targets whose exact candidate `src_path` changed enter the required change-owned set. Other eligible targets in selected packages enter the disjoint package-expansion set, which is executed only by the explicit final-candidate dispatch. Cargo `required-features` are sealed per eligible `package::target` in the digested candidate plan, and each shard activates the sorted union required by its package-scoped listing and execution commands. Missing, duplicate, or malformed feature rows invalidate the plan; a selected feature-gated target that lists no active tests remains a required failure. A push to `main` does not derive a test selection from the just-merged diff.

The candidate preflight classifies each `synchronize` event before toolchain
setup. An ordinary descendant push is a review repair. A merge whose non-first
parent is in the target history is a base merge; a rewritten series whose merge
base advances is a base rebase; another non-descendant update is a history
rewrite. Base updates require one exact
`Candidate-base-update: <head> <reason>` PR-body line, and other rewrites require
`Candidate-history-rewrite: <head> <reason>`. The declaration records necessity;
it does not replace review of a conflict resolution or approval for a force
push. Classifying a force-push needs the pre-push head, which is reachable
from no ref once it is replaced, so even a `fetch-depth: 0` checkout has to
fetch that commit by SHA before running the classifier. All three invocations
do, and none of them hides the result.

In the two detector workflows one step owns that clone's shape. It performs
the job's fetches and then asserts a stated property: every commit the three
verifiers read is present, and every pair they compare has a merge base the
clone can walk to. Verifiers depend on that property rather than on what an
earlier step's fetches happened to leave, which is what chelis#2228 and
chelis#2234 both were. A graft is what breaks the property and only a
deepening fetch removes one, so the escalation when the cheap shape does not
satisfy the assertion is `--unshallow`; the head deepen runs only against an
already shallow clone, because against a complete one it would create the
graft rather than avoid it. An invariant a later step could violate would
read as a guarantee it no longer gives, so that step owns every fetch in its
job and a test fails if another acquires one. The rule covers checkouts as
well as run lines: `actions/checkout` takes a fetch depth as an input and
re-clones, so it is the command that creates the graft in the first place
and never appears in a run line at all. A clone that cannot be
established fails the candidate preflight naming which pair it could not
resolve. An update that still cannot be classified fails closed, requires the
history-rewrite declaration, and the failure names that cause rather than
reporting a missing line as if the author had forgotten to write one.
The introducing implementation event requires the exact new head. The
acknowledgement workflow also runs on later body edits and accepts that recorded
head only while it remains an ancestor of the current head, so deleting a
previously required declaration cannot preserve a green required context.

That rerun is expected rather than incidental, since the contract asks for a
body edit recording the reviewed head immediately before merging. It and the
`Changelog` check therefore key their concurrency group by head SHA and do not
cancel in progress: a run for an older head answers about a different commit
and never races a newer one, while a second event on the same head is a
metadata change whose rerun would otherwise cancel the in-flight verdict and
leave a `cancelled` check run that reads as a failure. Both checks take about
half a minute, so letting both finish costs less than the misreading.
`Changelog` keeps its `edited` trigger because GitHub delivers a base-branch
change that way and the check reads `base.sha`. Narrowing that trigger with a
job-level condition is not available: a skipped required context satisfies this
repository's branch protection, so a body edit could turn a failing `Changelog`
into a passing one.

The same first-stage classifier identifies changes to workflows, workflow
actions, CI ownership, CI scripts and tests, `AGENTS.md`, this document, and the
PR-author guide. A workflow-native bootstrap applies the same conservative path
boundary independently, so narrowing the candidate-controlled classifier still
runs its contract tests. Those changes run the cheap routing, lifecycle,
change-owned topology and hosted-coverage unit suites in the detector job. A
failure is recorded as `candidate_preflight=failure` with a stated reason, and
the required Docs context states it: it fails naming what could not be
evaluated. On a code pull request two more required contexts go red with it,
`Lint and Unit Tests (Linux)` and `Integration Tests (Linux)`. Those two are
aggregators that do not gate on the verdict; their workers skip, and
`ci_require_success.py` fails on a skipped dependency, so they report the
dependency rather than the cause. Only on a documentation-only pull request,
where those workers are already out of scope, is Docs the only red context.

Three red contexts rather than one is more than a reader needs, and it is
deliberately not reduced further: making the aggregators skip would leave
Docs alone, and a skipped required context satisfies branch protection here,
so the count is also the margin. Hull is the one that was reduced, because
its red said nothing the others did not.

No step in that job can fail it, job setup aside: an unresolvable action
reference or a lost runner still fails it, and the required Docs context
reports that case. A hard failure there used to leave required
contexts with no check run at all, which cannot be waited out, re-run into
existence or overridden, and that is what made a pull request unmergeable
rather than merely red. The job therefore always completes and always emits
its outputs. Tolerating a step must not make its failure ignorable, so the
gate reads every one of them and a test fails if a tolerated step is not
read. The identity step runs between the gate and the verdict, behind the
same condition it had when it sat after the verdict, so moving it earlier
does not run it on a candidate the verdict would have stopped.

CI and Hull each publish an immutable identity for the synthetic candidate they
checked out. The checked-out candidate's first parent is the authoritative
target snapshot, even when `main` advances after GitHub creates the event
payload; its second parent must still equal the event's exact PR head. The
identity records those exact two parents and describes the PR patch from
`merge-base(base, head)..head`; using `base..head` would incorrectly count
unrelated target-branch advances as pull-request changes. Resolving that merge
base means walking both parents back to the branch point, so the detector job's
backstop fetches of the event base and of the candidate's first parent carry no
depth limit. A depth-limited fetch records its commit in `.git/shallow`, after
which git ignores the parents that commit already holds, and `merge-base` then
reports no common ancestor at all (chelis#2228). The identity script separates
that truncation from parents that genuinely share no history. It also records the
stable patch id, an exact normalized-diff digest that retains added and removed
bytes, and the changed paths and their digest. These producer artifacts are
candidate-controlled inputs, not receipts.

The default branch's `workflow_run` collector fetches the named Git objects
without checking out or executing pull-request content and recomputes every
identity field itself. It requires CI and Hull to name the same synthetic
candidate, reads the current branch-protection set, and binds every required
context to its expected workflow file and exact job id. An unknown required
context, a stale head, mismatched workflow provenance, missing artifact,
non-green latest check, or contradictory candidate identity withholds or fails
the receipt. The receipt also marks workflow, CI-policy, agent-contract and
CI-script changes as ineligible for later evidence reuse; their green state is
recorded, but candidate-controlled validation logic cannot authorize its own
reuse. A successful `pr-candidate-receipt-<head-sha>` artifact is therefore
trusted evidence keyed to the exact prior head for a later targeted-rebase
lane. On a `synchronize` event, CI and Hull check out the trusted verifier from
the exact target SHA and retrieve only the receipt named for `event.before`.
The verifier requires the same pull request and target, a strict forward base
advance, a new head containing that base, exact synthetic-candidate parents, no
retarget after the receipt, and a receipt marked eligible for reuse. It records
patch-identity changes and path overlap for review. The complete validation
delta runs from the receipt's prior synthetic candidate to the current
synthetic candidate, so it includes both the conflict resolution and target
movement included in that frozen candidate; `event.before..event.after` remains
the exact head-rewrite record. Before selecting an incremental lane, the trusted
verifier applies the planner's exact preflight classification to every delta
path, including Cargo metadata identities for integration targets and the
reviewed package/path rules.
An ambiguous target, path owned by standing evidence that the targeted lane
would reuse rather than rerun, CI-policy delta, stale or missing receipt,
retarget, non-forward update, candidate mismatch, unmapped path, or uncertain
history falls back to full CI. This includes inputs owned by `ci-fast`, which
the targeted lane otherwise skips. That fail-closed lane overrides the ordinary
docs-only skip. A linear content update remains on ordinary docs-aware routing
rather than attempting receipt reuse. No pull-request workflow can issue its
own trusted receipt.

The prior receipt already covers the earlier PR patch, so the cheap docs lane
requires only the complete prior-to-current synthetic-candidate delta to be
documentation-only; the PR itself may contain code. It runs the contract
preflight, PR acknowledgements, changelog policy, Docs, and inexpensive metadata
paths. A code-bearing delta is classified into exact packages and reviewed owner
jobs. Package seeds expand through reverse workspace dependencies. The targeted
lane runs package-scoped Clippy, formatting, default-feature library/binary
units and existing doctest owners for that package frontier, every eligible
integration target in it, and only the additional Python/script, SMT, backend,
diagnostic or Hull owners selected by the frontier. A Rust-policy owner with no
package frontier runs the complete `lint-and-unit` stage. This includes
same-file and same-line conflict resolutions and target movement after the
rebase. Required contexts still report on the rewritten head; reuse
short-circuits only work outside the interaction frontier and final-expansion
work whose trusted evidence remains applicable.
The collector issues receipts only for ordinary `pull_request` candidates. A
base-retarget candidate is validated by trusted dispatch and coordinator paths
that the receipt schema does not model, so its absent receipt remains a
full-validation fallback rather than reusable evidence.

`.config/ci-test-targets.toml` is the versioned ownership manifest for this surface. `standing_target` rows feed `ci-fast`; `target_exclusion` and `test_exclusion` rows name their exact alternative workflow, job, cadence, reason, and tracking issue; and `path_rule` rows assign shared paths to exact packages or an existing automated owner. Other prose paths use the existing docs-only classifier, including its executable-document exceptions; a new changelog fragment needs no manifest row. Package qualification is retained throughout, including execution, so equal target names in different packages cannot create a Cargo selector cross product.

An exact `manual_only_target` row keeps an all-ignored integration target in
required change-owned coverage. Its plan-bound execution mode lists ignored
tests, rejects the row if any default-enabled test appears, and runs the complete
ignored suite with the same target and per-test receipts. It is not an exclusion
and zero active tests do not count as success.

The Linux workspace worker executes as four shards of one
`--partition hash:${{ matrix.shard }}/4` selection rather than as a single run.
The selection is unchanged and still unfiltered. A hash partition hides nothing:
nextest assigns every listed test to exactly one partition, so the union of the
four shards is the whole selection. That is the distinction
`scripts/test_ci_cadence.py` now draws. It still rejects dropping
`--ignore-default-filter`, which really can hide a newly added target, and still
rejects the census returning to the workspace worker; it additionally rejects a
shard matrix that does not enumerate 1..M of the command's own `hash:N/M`, which
is the only way a partition can drop coverage. It no longer rejects partitioning
as such.

One unsharded run of this selection had never finished inside any budget
(chelis#1819), and a cancelled nextest run writes no JUnit at all, so the nightly
reported nothing rather than reporting a failure. Four shards each report a
verdict and upload their own `junit-linux-full-N`. The two steps that are not
part of the partition, the profile-coverage listing and the stdlib self-test
corpus, run on shard 1 alone, and shard 1 is also the single writer of the
`linux-workspace` cache. The profile-coverage listing compares like with like on
the `--ignore-default-filter` basis the shards run on; comparing it against a
listing that omitted `--profile`, and so silently inherited the `default`
profile's own exclusions, subtracted those tests from one side of the equality
only (chelis#1781).

`scripts/ci_test_targets.py` first builds workspace product libraries and binaries so cross-package `cargo_bin` users and generated-C tests have their CLI and runtime static library. It lists all units and globally unique standing test names together, then lists shared names with exact package selectors. The combined binary receipt must contain every unit (including zero-test unit binaries) and exactly the selected integrations before any group executes. Each group runs once without default filters; command, listing, timing, and JUnit receipts preserve the whole union. The worker also writes a digested standing-coverage record binding the candidate SHA, normalized ownership configuration, exact execution mode, selected/executed targets, and per-test results. Missing unit binaries, duplicate or unexpected integration binaries, empty selected integrations, filtered non-ignored tests, incomplete results, or missing JUnit fail the worker. Ignored tests remain ignored; this pass neither deletes tests nor changes language semantics.

On a pull request or trusted exact-candidate dispatch, `Integration Tests (Linux)` fails closed on both the standing fast worker and the required change-owned report. The four required shards and the four package-expansion shards use deterministic longest-processing-time assignment from the reviewed `.config/ci-change-owned-durations.json` target baseline; unknown targets receive a conservative 30-second estimate. Each plan binds the exact weights, baseline digest and estimated shard totals. To remain dispatchable before a planner change merges, the v3 package-expansion hash fields remain a compatibility envelope for the trusted validator; a digested execution disposition owns the duration-balanced shards that candidate workers and reports use, and both representations must cover the same targets exactly. Refresh the baseline explicitly with `scripts/ci_change_owned.py build-duration-baseline` and authenticated plan/receipt artifact pairs; workspace-build time is excluded because every nonempty shard pays it independently. Estimate error is diagnostic and never changes exact coverage or test verdicts. The report rejects missing shards, digest disagreement, duplicate execution, uncovered selected targets, executed exclusions, and test failure. A change-owned target already in the standing set is removed from shard execution only when the report verifies the ci-fast record against the same candidate SHA, normalized configuration digest, exact execution mode, complete standing target set, and matching selected/executed per-test results. Missing, stale, partial, failed, or tampered standing evidence does not satisfy the obligation. The `integration-change-plan`, per-shard receipts, standing coverage, required report, JUnit, commands, selected/executed lists, and timings are retained for 14 days.

For an accepted targeted rebase, the planner compares the prior receipt's
synthetic candidate with the current synthetic candidate. It maps every code
path in that exact delta to an exact package or reviewed owner, expands package
seeds through reverse workspace dependencies, and makes every eligible
integration target in that package frontier required change-owned coverage.
The plan deliberately reuses no standing coverage receipt: its selected shards
execute and report the affected targets on the current synthetic candidate
before the required integration context passes.

On a push to `main`, `Integration Tests (Linux)` instead requires only the fixed `ci-fast` standing receipt. The planner, change-owned workers, and their report are skipped. This makes every default-branch commit answer the same standing acceptance question: a merge cannot make `main` red merely because its file diff happens to select known nightly residuals, and a later unrelated merge cannot make `main` green by selecting a different target set. Full workspace and hardware-sensitive residual work remains owned by the scheduled suites and its tracking issues.

Each worker downloads the current run's plan after Cargo cache restoration and before execution. The cache may replace `target/`, so it cannot own the plan; a missing artifact remains a job failure.
Before installing build dependencies or restoring that cache, a system-Python
step reads the digested plan. A plan-proven empty shard writes its successful
zero-target/zero-test receipt immediately and skips the build environment.
Nonempty shards retain the post-cache plan download and normal executor.

Package expansion uses a separate manually dispatched four-shard worker pool
after the review rounds settle the intended content; intermediate review
candidates do not dispatch it.
The dispatcher accepts an open pull request number and an exact expected head
SHA, validates both before planning, and rejects stale candidates. The planner
records the exact synthetic merge SHA and target branch name; every worker and
the summary check out that frozen SHA. The final validator requires the same PR
head and target branch name and verifies the frozen merge's exact planned
parents. Later movement of that same target branch therefore does not invalidate
the run, while a changed head or target retarget does. Classified failures,
unrun coverage, missing shards and exclusions are recorded by `Manual Package
Expansion Summary`; neither that summary nor its workers feed `Integration Tests
(Linux)`.
Once review repairs have fixed the intended content, agents start it alongside
the final required implementation checks; there is no dependency between their
verdicts. Inspect both before merging and record the reviewed SHA and run link.
An ordinary content change after expansion, a base-branch retarget, or a rebase
the trusted verifier does not accept requires a fresh dispatch. An accepted
rebase, including a hand-resolved conflict, retains the successful expansion on
the prior-receipt head after its complete delta-selected coverage passes and the
standing reviewer inspects the resolution. Record that expansion's SHA and run
together with the new rebase decision and required check run. Rebasing before
the first push creates no expansion evidence to invalidate. Do not rebase a
ready pull request merely because `main` advanced: if the exact reviewed head
still merges safely, preserve it and its evidence.
Introduced failures are resolved; inherited failures and incomplete coverage are
named explicitly. Nightly JUnit reports stay in their producing workflow.
The Linux nightly report inspects every execution worker and opens a failure
tracker on non-success; a manual branch run cannot close a main-nightly tracker.

The expansion summary makes that distinction itself rather than leaving it to a
reader. Before summarizing, the job records a default-branch failure baseline
with `scripts/ci_failure_baseline.py`: the newest completed `heavy-e2e.yml` run
on `main` that still retains all four `junit-linux-full-*` artifacts *and whose
commit the candidate's own merge base contains*. A red nightly qualifies,
because the redness is the evidence; a run missing a shard does not, because a
partial baseline reports that shard's inherited failures as introduced; and a
run the base does not contain does not, because the report would refuse it.
That last condition is not hypothetical: GitHub does not recompute a pull
request's merge ref as `main` advances, so an older candidate's base routinely
predates the newest nightly. The report then splits every observed failure into **introduced**
(the baseline ran that test and it passed, or the baseline never ran it, which
the row records as `absent`) and **inherited** (the baseline ran it and it
failed), and counts every selected target with no execution evidence as
**unrun**. The three counts are disjoint and none absorbs another: coverage the
lane could not reach is never reported as inherited. The report is clean when
nothing was introduced.

Two conditions fail the report loudly rather than differencing against the
wrong tree. A baseline that is absent, unreadable, or records no executed case
leaves the run with no classification at all, which is reported as such and is
not a clean result. A baseline commit the candidate's own merge base does not
contain is refused outright, because a baseline ahead of the candidate, or on
another branch, can report a failure the candidate introduced as one it
inherited.

Distance behind the base is annotated rather than refused, and it is not
harmless in one direction only. A test that goes red on `main` after the
baseline commit is reported here as introduced, which over-reports and is the
safe error. A test that was *failing* at the baseline, was fixed on `main`
since, and is broken again by the candidate keeps its identity in the
baseline's failing set and is therefore reported as inherited, which
under-reports. Twelve identities moved that way between two nightlies five
days apart, so the window is real. Selecting the newest baseline the base
contains makes that window the commits between the baseline and the base and
no larger; closing it entirely would mean running the suite on the base
itself. The summary states the baseline's run, commit, timestamp and distance
in commits so a reader can size the residual, and says which direction each
error runs in.

There is no soft-budget finding. The per-shard estimate is a
longest-processing-time balancing weight derived from serial per-target
measurements, while the executor runs up to sixteen targets of one package in a
single command, so it overstates a completed shard and understates one the
deadline cut. Measured over every dispatch since the lane existed, every shard
that exhausted the hard deadline had also exceeded the soft budget, and every
shard that exceeded the budget without the deadline had executed its complete
selection, so the comparison never reported a defect of its own. The summary
prints each shard's elapsed time and its balancing weight, and attaches no
verdict to either.

The expansion executor splits ordinary targets into bounded exact-package groups, then issues one list command and one run command per group. The grouping limit changes command boundaries only: every selected target remains required in the informational receipt. Manual-only targets and targets with exact test exclusions remain singleton commands so ignored-test mode and filters cannot affect siblings. An ordinary expansion target whose listing contains no applicable non-ignored test is recorded as inspected and not applicable; an entirely inapplicable group does not launch a guaranteed-empty nextest run. Required change-owned targets remain fail-closed when no active test applies. Listings and JUnit are decomposed back into exact `package::target::test` evidence; a missing applicable result leaves that target incomplete and the shard unsuccessful. The executor stops after a shared 16-minute build/list/test budget and writes an unsuccessful receipt containing completed-group evidence and the complete selected-target list. It terminates the active command's process group and starts no further command. Unfinished coverage is reported as the informational summary's unrun count rather than as a finding. The 20-minute job limit remains; the intervening time allows upload after executor expiry. The receipt still records `soft_budget_exceeded` as a measurement, and nothing reads it as a verdict. Every nonempty shard's workspace-product build pins `CHELIS_RUNTIME_LIB` to its exact-head `libchelis_runtime.a` before target listing or execution, so fixtures cannot start a nested or stale Cargo build. Setup delays, external cancellation or runner loss can still prevent a receipt. Required change-owned execution retains its existing limit and remains fail-closed on incomplete coverage.

Use `gh workflow run heavy-e2e.yml --ref BRANCH` or `gh workflow run macos-nightly.yml --ref BRANCH` for candidate validation. Full Linux execution shards, dtype, script integrations and generalization shards have 60-minute timeouts; other extended Linux workers have 45 minutes, and Darwin SMT has 60. Dtype previously exhausted 45 minutes; its 60-minute allowance preserves complete execution while census work is removed from the other Linux workers. Existing ignored/manual gates still require their documented prerequisite and explicit invocation. The stdlib self-test corpus remains explicitly invoked nightly. Existing nightly failures must be recorded against a baseline, never treated as passing evidence.

The developer's `gate.py --fast`, `--local`, `integration`, and full/manual commands retain their previous selections. The separate `gate.py ci-fast` stage owns the fixed standing hosted selection on pull requests and `main`. The required change-owned lane runs only for PR candidates and trusted exact-candidate dispatches; package expansion runs only through its explicit PR dispatch. A title or description edit reruns the dedicated acknowledgement check and changelog policy without entering compiler or Hull workflows, so the unchanged candidate's implementation contexts are neither cancelled nor replaced. A base retarget creates a required pending head receipt, validates the open PR's exact head/base and the checked-out two-parent merge, dispatches CI and Hull from the trusted new base, and closes the receipt only after both runs succeed. Run the owning phase oracle on the candidate when claiming phase completion.

### Measured figures for the changed-path classification stage

Taken 2026-09-20 against `64a446998` on an Apple-silicon workstation. A figure
here is evidence with a date on it, not a constant: re-run the command before
relying on one, and correct this table rather than the prose that cites it.

| figure | conditions | how it was taken |
|---|---|---|
| 0.11-0.13s wall | any non-empty change set, `--from-git` | `/usr/bin/time -p .venv/bin/python scripts/ci_change_owned.py classify-paths --from-git`, five runs |
| 0.05-0.06s wall | empty change set, which returns before `cargo metadata` | same command on an unchanged tree |
| `cargo metadata --no-deps --locked` 0.02-0.04s | warm, repeated invocation | the stage's dominant cost; 0.24s with dependencies, which it does not ask for |

**The cost does not split by warmth, and that is the measured result rather
than an omission.** Cold cargo without a bytecode cache, warm cargo after a
real `cargo fmt --all` without a bytecode cache, and warm cargo with the
cache all came out the same to within noise. `gate.py` sets no
`PYTHONDONTWRITEBYTECODE` for its children and `cargo fmt` runs before this
stage, so there were four plausibly distinct quantities here and the
measurement found one. Recorded so the next reader does not re-derive the
hypothesis: only the empty set differs.

The form does matter, which is the distinction that survived. Passing paths
as arguments costs about 0.12s of CPU and `--from-git` about 0.18s, the
difference being two `git diff` forks. `--from-git` is the shipped form and
the figures above are its.

## Python execution ownership and timing

`python scripts/ci_script_tests.py pr` runs the cheap discovered tests; `nightly`
runs the compiler-dependent classes listed in that script. The selection receipt
assigns every discovered test to exactly one of PR, script nightly, census or
profile-oracle ownership. Classes absent from discovery and absent census control
identities fail selection. New methods inherit their class's cadence; ordinary
new classes enter the PR selection, whose job is bounded to ten minutes.

Heavy methods already present in the wire/binding oracle's exact Python selections
are owned there and omitted from script nightly. Profile listing classes retain
their workspace/generalization owners. Direct `unittest discover` and local/full
gate commands remain exhaustive. To run the heavy Python checks manually use
`.venv/bin/python scripts/ci_script_tests.py nightly`; complete census acceptance
still requires `.venv/bin/python scripts/dtype_phase3_oracle.py`.

The stdlib generator's real determinism test regenerates twice and exercises the
production stale/unstable-output comparison against that fresh pair. Its negative
controls and output-restoration tests remain, while a second pair of full builds
is removed. Restored Cargo artifacts accelerate compilation; they never substitute
for executing tests or create a numeric authority witness.

The `script-tests-pr`, `script-tests-nightly` and `script-tests-census` artifacts
retain selection and per-process JSONL timings for 14 days, including failed runs.
Each subprocess has a start record and a finish record with elapsed seconds;
unfinished starts identify the command active at cancellation. Test and class-setup
timings identify costs hidden in `setUpClass`. Set `CHELIS_CI_TIMING_DIR` to an
absolute directory to enable census subprocess diagnostics locally. Timing data
is diagnostic only and carries no correctness authority.
