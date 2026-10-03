# CI validation cadence

For the actions to take when preparing a PR, read
[Changing tests, inventories and protected contracts](guard_changes_for_pr_authors.md).
[The local gate](local_gate.md) records what `scripts/gate.py` runs locally.

Ordinary PRs and main pushes use Linux. Passing required PR checks is **not a phase acceptance result** for a full or feature-specific oracle that runs nightly.

| Owner | Cadence | Coverage |
|---|---|---|
| `ci.yml` / `conformance.yml` candidate preflight | Every PR implementation event; CI-contract tests only when their path classifier fires | Rejects undeclared base merges, base-changing rebases and other history rewrites; runs bootstrap-light workflow/routing tests before expensive build fan-out |
| `ci.yml` `ci-fast` | Ordinary PR and main push, with the existing docs-only skip; accepted targeted rebases use the interaction frontier instead | Every default-feature library/binary unit target and the reviewed `standing_target` identities in `.config/ci-test-targets.toml`, each with its Cargo-declared required features activated; 55-minute total job timeout |
| `ci.yml` change-owned shards and report | Ordinary PR and trusted exact-candidate workflow dispatch, with the existing docs-only skip; accepted targeted rebases run only when their package frontier selects integration coverage | Every integration target added or directly modified by the candidate, with its Cargo-declared required features activated, or its exact reviewed alternative owner; targeted rebases require every eligible target in affected packages and their reverse workspace dependents; four deterministic shards with an 85-minute total job timeout each |
| `pr-package-expansion.yml` | Manual dispatch after review repairs, parallel with final required checks, using an open PR number and exact expected head SHA | Other integration targets in directly selected packages, with their Cargo-declared required features activated and exact reviewed target/test rows excluded; four informational shards with an 80-minute execution deadline inside a 90-minute job limit and a separate summary that classifies every observed failure as introduced or inherited against the nightly default-branch baseline and counts unrun coverage on its own |
| `pr-contract-acknowledgements.yml` | PR open, synchronize, reopen, title/body edit or base retarget | Dedicated required validation of persistent candidate-lifecycle, protected-test and frozen-contract acknowledgement lines; no compiler build |
| `pr-base-retarget.yml` | PR open/synchronize/reopen plus base-retarget coordination | Required head receipt. Ordinary candidates defer to the normal required implementation contexts. A base retarget holds the head pending while trusted-base coordination dispatches exact-head/exact-base CI and Hull runs against the new synthetic merge |
| `pr-candidate-receipt.yml` | Completion of any workflow that can finish the required PR check set | Default-branch-owned receipt binding the exact PR head, synthetic candidate, patch identity, required check runs and their workflow/job provenance; eligible receipts can authorize the guarded targeted-rebase lane |
| `ci.yml` retained workers | Ordinary PR and main push; accepted targeted rebases run only the owners selected by their interaction frontier | Rust policy and doctests, Python/script units selected by `ci_script_tests.py pr`, focused SMT plus its existing Deep-obligation integration target, Linux glibc compatibility, Docs, backend sanitizer units and explicit backend doctests; change-triggered diagnostic mutation and the offline rejection-authority boundary |
| `conformance.yml` | PR and main push | Existing frozen Hull conformance gate |
| `heavy-e2e.yml` | Daily 03:17 UTC full run; manual `all` (default), `runtime-representation`, `dtype-phase3`, or `runtime-extent` | Full runs cover the non-ignored default workspace across four shards, the dtype owner, script integrations, exhaustive generalization partitions, phase oracles, frontend/domain support, and backend sanitizers. Each scoped dispatch starts its named oracle and a scope-receipt job. |
| `macos-nightly.yml` | Daily 04:17 UTC and manual dispatch | Both Mac workspace partitions, both Clippy configurations, the ownership-ledger targets, architecture/ABI/Metal smokes, and Darwin SMT |
| `build-cvc5.yml` Darwin producer | Daily 01:17 UTC and manual dispatch | Missing Darwin prebuilt assets; relevant main pushes may produce Linux assets only |
| `smt-full-prove.yml` | Existing nightly/manual cadence | Full SMT proof validation |
| `release.yml` and release-triggered Nix checks | Release/manual cadence | Existing shipping-artifact validation, including Mac |

Migrated Devenv-backed jobs reserve cold setup headroom without changing execution commands or script deadlines. The total job timeout starts at job launch and includes setup, execution and artifact upload; it is not a separate script execution deadline. Thus `ci-fast` retains its prior 30-minute budget within a 55-minute job timeout, and required change-owned shards retain their prior 60-minute budget within an 85-minute job timeout. Native Nix validation has total job timeouts of 135 minutes on Linux and 145 minutes on Darwin. The manual package-expansion lane remains separate: its executor's 80-minute deadline starts at `run-shard`, inside the unchanged 90-minute total job timeout.

Rust policy and the native SMT feature lane retain total job timeouts of 75 and 90 minutes respectively. These are cold-build safety limits, not a measured performance result.

GitHub-hosted runners run every job for ordinary pull-request and push events. `Fast Tests (Linux)` stays hosted because on the warm pool it would queue behind the nightly, and the pool host admits pull requests only against `main`. Explicit dispatches keep their existing route: `execution_target=self-hosted` requires an exact candidate SHA and a matching reviewed route on the runner host, and `github-hosted` dispatches stay hosted. The workflows grant no runner routing for ordinary events.

In a full `heavy-e2e.yml` run on `main`, every execution job except `test-telemetry` uses the `chelis-ci-warm-x64` label. The separate dispatch-scope receipt job uses a GitHub-hosted runner. Dispatches on any other branch stay GitHub-hosted. GitHub Actions has no job priority, so on `main` the warm jobs share the `linux-extended-warm-x64` concurrency group with `queue: max`: the nightly holds at most one runner of the two-runner pool and leaves the other free for the pool's other jobs (chelis#2543). Waiting jobs stay pending in order instead of being cancelled, so a `main` run lasts about the sum of its jobs. Off `main` each job has its own group, so candidate validation stays parallel. `scripts/test_ci_cadence.py` locks both properties.

Eligible jobs execute repository commands through `chelis-ci-shell run` and `chelis-gate` whichever runner kind executes them. `.github/actions/setup-project-ci` supplies both. GitHub-hosted runners cannot reach the private Nix cache, so they keep the toolchain main uses: Ubuntu's C compilers and OpenBLAS, a stable rustup toolchain, the uv-managed interpreter from `scripts/ci_setup_uv_python.py`, and two plain shims from `scripts/ci_hosted_commands.py`; Cargo compiles the numerics from their vendored sources and no Devenv profile is realized there. The self-hosted runner activates the `ci` profile once, which owns the Nix-built numerics cache and the static LP64 OpenBLAS provider substituted from the private cache; its SMT worker selects `ci-smt`, which extends `ci` with the CVC5 closure. Python, its virtual environment and PyO3 use one project-owned interpreter on either runner. Native library paths apply to repository commands, not GitHub action runtimes.

The self-hosted CI profile owns GNU Make's `make` and `gmake` entry points, the LP64 OpenBLAS ABI and Linux Fortran provider, and the native libraries needed by embedded Python/NumPy. Compiler wrappers preserve arbitrary path bytes; Linux linkers emit content-derived GNU build IDs. `.kache.toml` includes these compiler/linker policy variables in artifact keys. Self-hosted runners rely on Kache and skip the GitHub Rust target caches, so a Nix-built output never reaches a hosted job.

CI, Hull and package expansion grant every job read-only GitHub dependency-cache access with `cache-mode: read`, including dispatches that execute a PR candidate in the base branch context. GitHub enforces that capability separately from `GITHUB_TOKEN` permissions, and the key takes only a literal value, so these workflows restore GitHub caches and save none, pushes to `main` included. `ci-cache-warm.yml` writes the caches they restore; [Rust build caches](#rust-build-caches) describes the policy. This setting controls GitHub caches; the private Nix/Kache publication policy below has its own admission boundary. Candidate identity and retarget steps pass the event's base branch name through a quoted environment variable so it is an argument rather than shell source. [GitHub's cache access reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#controlling-cache-access-with-cache-mode) describes the enforced modes.

Self-hosted Linux C dependencies use the Nix GCC provider in GNU C17 mode with PIC; the vendored GMP/MPFR configure checks fail with the pinned Clang C provider. Shell entry reasserts that selection after compiler package setup hooks, which can overwrite `CC`. C++ and Rust retain their declared Devenv compiler/linker selections.

`ci.yml` and Hull use `.github/actions/setup-project-ci` to order cache setup. On a self-hosted worker it restores signed Nix substitutions **before** project realization, attaches the resulting repository-owned Kache executable to the same job state, and declares the realized profile plus any environment-only `CVC5_DIR` tree. Every caller finalizes nonempty state with `always()`, including after failed activation or attachment. Push and dispatch workers request read/write, including an explicitly reviewed draft: a branch check must not prevent cache warming. Pull-request workers request read only, so a dependency's build script running in a pull-request job cannot publish what every other worker substitutes; merged and reviewed candidates warm the caches. This is publication intent, not an admission grant. Tunnet membership remains the cache-access boundary; the action does not acquire OIDC or AWS credentials. Hosted workers restore the GitHub Rust build caches instead and do not contact the private caches.

`.github/actions/vendor/ci/` holds byte-for-byte copies of the five `Chelis-Lang/ci` actions these workflows call (`setup-devenv`, `reclaim-ubuntu-runner-disk`, `openspec-store`, `restore-build-cache`, and `publish-build-cache`) at `6afc09f65c178211a9d1bea994a4b39b62d23024`, because a public repository cannot resolve actions from a private one (chelis#2867). The copy keeps that repository's layout and includes the paths the actions read outside their own directories: `devenv/shared/`, `devenv-toolchain.toml`, the `actions/openspec-governance/` npm lock, and the `tools/` build-cache scripts. Their tests stay in `Chelis-Lang/ci`, and `chelis-lint.toml` excludes the copy as a vendored dependency because the `setup-devenv` bootstrap is shell. To update the copy, replace the same paths from a newer reviewed revision and record it here.

The self-hosted CI profile disables Cargo incremental compilation for shared compiler caching and omits the legacy Kache schema fixture. That fixture remains in the default development profile for its dedicated transition oracle. Self-hosted Linux Rust policy selects `lint-and-unit-nix`: it preserves every existing policy command and solver-free feature while adding `chelis-prove/ci-openblas-system`, which selects the pinned static LP64 OpenBLAS archive through pkg-config rather than rebuilding it in Cargo. Hosted Rust policy runs `lint-and-unit` with the bundled provider, as main does; the developer gate retains its bundled provider and Darwin retains Accelerate.

The `ci-smt` provider binds CVC5 1.3.1 to the locked cvc5-sys 0.3.1 checksum, its Production configuration, parser, CaDiCaL and LibPoly dependencies. Cargo consumes its `CVC5_DIR` on the self-hosted worker; the hosted worker fetches, harvests and republishes the prebuilt cvc5 stores through `scripts/ci_cvc5_cache.py`. The glibc 2.31 compatibility lane, the native release producers and the frozen manual `nix-packages.yml` recipe keep their own toolchains.

The runtime-representation inventory admits Clang's canonical resource `include` directory, including split Nix installations where it is a symlink. Resource-root siblings and headers escaping that include directory remain outside the inventory universe. Its fixed-target scan disables host C++ standard-library headers and uses the existing SDK stubs.

The Linux workspace worker passes `--ignore-default-filter` and deliberately includes the PR selection. Only `chelis-compiler-api::capacity_census_wire` and `chelis-python::capacity_census_bindings` are excluded: the dtype worker executes both. The generalization worker uses the same census exclusions; census authority is checked on the default-feature configuration. Executable listing set-math proves that workspace plus dtype still covers every non-ignored test in the unfiltered corpus, with none in neither selection and no census test in both. The two selections do overlap elsewhere: the dtype oracle also owns several non-census binaries. A featureless workspace run skips every target whose Cargo `required-features` are not default-enabled. The `ownership-ledger` targets therefore run with that feature as the gate's own integration commands: in the `integration-support` slices on Linux, and on macOS in `macos-nightly.yml`'s `macos-ownership-ledger` job, which `macos-smoke` requires. Each command runs `scripts/ownership_ledger_tests.py` for one package, which reads the package's targets that require `ownership-ledger` from Cargo metadata, so neither the gate nor the workflow repeats the list. The `chelis-compiler-api` command runs in the frontend slice. The `chelis-cli` command runs last in the domain slice, because it rebuilds `target/debug/chelis` against the instrumented runtime, and the gate hands that path to the unrepresentable-domain oracle when an earlier command built it.

The regular candidate planner runs for pull requests and trusted exact-candidate workflow dispatches. It requires the checked-out synthetic merge commit to have exactly two parents and requires the event head to be `HEAD^2`. Its rename-aware NUL-delimited diff assigns every changed path one final disposition. Package roots map through Cargo metadata, reviewed shared paths map through `path_rule` rows, and unknown, stale, duplicate, or ambiguous mappings fail planning. Added targets and targets whose exact candidate `src_path` changed enter the required change-owned set. Other eligible targets in selected packages enter the disjoint package-expansion set, which is executed only by the explicit final-candidate dispatch. Cargo `required-features` are sealed per eligible `package::target` in the digested candidate plan, and each listing and execution command activates exactly the sorted features its targets require: change-owned targets run alone, and a package-expansion chunk holds only targets of one package with the same required-feature set. Missing, duplicate, or malformed feature rows invalidate the plan; a selected feature-gated target that lists no active tests remains a required failure. A push to `main` does not derive a test selection from the just-merged diff.

The candidate preflight classifies each `synchronize` event before toolchain
setup. An ordinary descendant push is a review repair. A merge whose non-first
parent is in the target history is a base merge; a rewritten series whose merge
base advances is a base rebase; another non-descendant update is a history
rewrite. Base updates require one exact
`Candidate-base-update: <head> <reason>` PR-body line, and other rewrites require
`Candidate-history-rewrite: <head> <reason>`. The declaration records necessity;
it does not replace review of a conflict resolution. Classifying a force-push needs the pre-push head, which is reachable
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
reports that case. A hard failure there would leave required
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
the receipt. The one exception to "latest" is Secret scan, which also runs on
every push: its push-event run on the same head is passed over, and its latest
pull_request run is the evidence. The receipt also marks workflow, CI-policy, agent-contract and
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

A test exclusion moves a complete check off pull-request CI without weakening what that check proves. Where pull-request CI keeps a smaller canary of an excluded test in the same target, as for the Std.Decimal differential corpus and extreme-argument sweep (chelis#2794), a green pull request shows only that the canary's named inputs pass. The excluded test runs in its owner job, here the nightly `heavy-e2e.yml` `module-oracles` job, which runs every test exclusion that `.config/ci-test-targets.toml` assigns to it while the full-workspace shards leave those tests out, in the macOS nightly, and on demand through `docs/manual_gates.md`; until one of those runs passes on a commit that contains a change, the complete check has not run for it.

An exact `manual_only_target` row keeps an all-ignored integration target in
required change-owned coverage. Its plan-bound execution mode lists ignored
tests, rejects the row if any default-enabled test appears, and runs the complete
ignored suite with the same target and per-test receipts. It is not an exclusion
and zero active tests do not count as success. A `manual_gate_target` row takes
the same listing and rejection for a target whose prerequisites no Linux PR
worker has, cites the `docs/manual_gates.md` wired-gate rows whose commands run
exactly that target's ignored tests, one of them with no name filter, runs none
of its tests, and appears in the report as a manual gate not executed in PR CI
rather than as a covered target.

Test-authored C fixtures, the C a test writes and compiles with `cc`, are
portability tests whether or not they were written as one (chelis#2496).
Authors run them on macOS, where Apple clang accepts both defects that
chelis#1864 merged: a `PRIu64` whose `<inttypes.h>` arrived only through the
Accelerate framework that `chelis_math.h` includes on Apple, and an unbraced
`if` followed by a second statement on its line, which GCC's `-Wall` warns about
and `-Werror` makes an error. A local run, `gate.py --validation` included, is
therefore no portability evidence on macOS. Before merge a fixture reaches Linux
only through the existing selection and nothing wider. An added or directly
modified fixture target runs in the required change-owned lane with its Cargo
`required-features` activated. A change to a shared fixture
helper, such as `crates/chelis-compiler-api/tests/ownership_support/`, reaches
its standing dependents in `ci-fast` and the rest only through the
package-expansion dispatch that precedes merge.
Each fixture's own compiler choice and flags decide what fails it; one that must
reject GCC warnings opts into `-std=c11 -Wall -Wextra -Werror`. The standing
`chelis-runtime::c_fixture_portability` target checks the toolchain it runs on:
on Linux it fails unless `cc` is GCC and those flags reject both chelis#1864
shapes while accepting their repaired twins. A lane that does not run it gets no
evidence from it. Making every fixture target standing would be the
whole-workspace pull-request suite whose cost chelis#1824 weighs;
generated-source portability as a class stays with chelis#2063.

The Linux workspace worker executes as four shards of one
`--partition hash:${{ matrix.shard }}/4` selection rather than as a single run.
The selection is unchanged and still unfiltered. A hash partition hides nothing:
nextest assigns every listed test to exactly one partition, so the union of the
four shards is the whole selection. That is the distinction
`scripts/test_ci_cadence.py` draws. It rejects dropping
`--ignore-default-filter`, which really can hide a newly added target, and
rejects the census returning to the workspace worker; it also rejects a
shard matrix that does not enumerate 1..M of the command's own `hash:N/M`, which
is the only way a partition can drop coverage. It does not reject partitioning
as such.

An unsharded run of this selection does not finish inside any budget, and a cancelled nextest run writes no JUnit at all, so the nightly
reported nothing rather than reporting a failure. Four shards each report a
verdict and upload their own `junit-linux-full-N`. The two steps that are not
part of the partition, the profile-coverage listing and the stdlib self-test
corpus, run on shard 1 alone. On a hosted run every shard restores the
`linux-workspace` cache that `ci-cache-warm.yml` writes. The profile-coverage listing compares like with like on
the `--ignore-default-filter` basis the shards run on; comparing it against a
listing that omitted `--profile`, and so silently inherited the `default`
profile's own exclusions, subtracted those tests from one side of the equality
only (chelis#1781).

`scripts/ci_test_targets.py` first builds workspace product libraries and binaries so cross-package `cargo_bin` users and generated-C tests have their CLI and runtime static library. It lists all units and globally unique standing test names together, then lists shared names with exact package selectors. A standing target whose Cargo `required-features` are not all default-enabled is listed last, in one group per package and exact feature set (`-p <package> --features <features> --test ...`), so no other unit compiles with those features; its features come from Cargo metadata, not from the manifest row. The combined binary receipt must contain every unit (including zero-test unit binaries) and exactly the selected integrations before any group executes. Each group runs once without default filters; command, listing, timing, and JUnit receipts preserve the whole union. The worker also writes a digested standing-coverage record binding the candidate SHA, normalized ownership configuration, exact execution mode, selected/executed targets, and per-test results. Missing unit binaries, duplicate or unexpected integration binaries, empty selected integrations, filtered non-ignored tests, incomplete results, or missing JUnit fail the worker. Ignored tests remain ignored; this pass neither deletes tests nor changes language semantics.

On a pull request or trusted exact-candidate dispatch, `Integration Tests (Linux)` fails closed on both the standing fast worker and the required change-owned report. The four required shards and the four package-expansion shards use deterministic longest-processing-time assignment from the reviewed `.config/ci-change-owned-durations.json` target baseline; unknown targets receive a 30-second fallback that may underestimate a new long test. Each plan binds the exact weights, baseline digest and estimated shard totals. To remain dispatchable before a planner change merges, the v3 package-expansion hash fields remain a compatibility envelope for the trusted validator; a digested execution disposition owns the duration-balanced shards that candidate workers and reports use, and both representations must cover the same targets exactly. Refresh the baseline explicitly with `scripts/ci_change_owned.py build-duration-baseline` and authenticated plan/receipt artifact pairs; `--seed` retains a reviewed prior baseline. The builder accepts complete successful expansion receipts as well as change-owned receipts. For a grouped expansion command it divides observed command wall time among targets in proportion to their JUnit test times, with each target's longest test as a floor. This is a scheduling estimate, not an exact per-target runtime measurement. Workspace-build time is excluded because every nonempty shard pays it independently. Estimate error is diagnostic and never changes exact coverage or test verdicts. The report rejects missing shards, digest disagreement, duplicate execution, uncovered selected targets, executed exclusions, and test failure. A change-owned target already in the standing set is removed from shard execution only when the report verifies the ci-fast record against the same candidate SHA, normalized configuration digest, exact execution mode, complete standing target set, and matching selected/executed per-test results. Missing, stale, partial, failed, or tampered standing evidence does not satisfy the obligation. The `integration-change-plan`, per-shard receipts, standing coverage, required report, JUnit, commands, selected/executed lists, and timings are retained for 14 days.

The baseline combines authenticated plan and receipt artifacts from complete package-expansion runs with change-owned sources. Replaying a run's measured target work through the refreshed assignment, with its observed workspace-build durations, gives a modelled schedule rather than an observed run: grouping, cache state and nested census builds can change it, so the next complete dispatch after a refresh is its acceptance measurement. Random assignment by whole package would leave the large `chelis-cli` package indivisible, and the four independent hosted jobs have no shared work queue to support work stealing.

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
with `scripts/ci_failure_baseline.py`: the newest eligible `heavy-e2e.yml`
schedule or full-scope dispatch on `main` among the ten most recent runs; when
that window has no usable baseline, the selector checks the ten most recent
scheduled runs. A baseline must retain all four `junit-linux-full-*` artifacts,
and the candidate's own merge base must contain its commit. A dispatch also
needs its unique `linux-extended-dispatch-scope`
artifact to say `all` and bind the exact run ID and head SHA; a scoped dispatch
cannot supply a baseline even if it retains four JUnits. Schedules from before
the scope input remain usable without a receipt. A red nightly qualifies,
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
deadline cut. That is why no budget compared against it can carry information,
and it is unaffected by where the deadline sits. The summary prints each shard's
elapsed time and its balancing weight, and attaches no verdict to either.

The soft budget survives as an early warning and nothing more: it is derived as
ten minutes below the execution deadline rather than set independently, so a
receipt records `soft_budget_exceeded` exactly when that shard came within ten
minutes of being cut. The flag is computed once from total elapsed, so a shard
the deadline cut always sets it; its only discriminating power is over shards
that finished. The window is ten minutes rather than the permitted minimum
because elapsed time advances in whole commands, and a window narrower than one
command is jumped over rather than landed in: the longest single command
observed is about 593s, so 600s clears it by seven seconds. Deriving the budget
from the deadline, rather than setting it as an independent constant, keeps the
pair from drifting into a budget every shard exceeds, which would warn nobody.

### Measured figures for the expansion deadline

Every number the prose, the comments and the tests' docstrings above rely on,
with what it measures and how to re-take it. A figure here is a measurement, not a
constant: re-run the command before relying on one, and correct this table
rather than the prose that cites it.

| figure | what it measures | how it was taken |
|---|---|---|
| setup 118s median, 163s p90, 211s max | job start to the executor's first command, which is subtracted from the gap between the deadline and the job limit | step timestamps of the 104 shard jobs in the last 30 `pr-package-expansion.yml` dispatches, via `gh api repos/Chelis-Lang/chelis/actions/runs/<id>/jobs`. Every sampled job had a warm `rust-cache` hit; under `save-if: false` a miss downloads nothing, so a miss makes setup faster rather than slower |
| post-executor 11s max | receipt upload and post-steps, the other side of the gap | same sample |
| finalization 0.28s, and 1.58s at 98 MB | merging, digesting and writing a shard's JUnit and three sidecars, which the gap does *not* need to cover | probe over 104 targets x 500 cases (25.6 MB) and 200 x 1000 (98 MB) |
| real slack 18s at 16-in-20, 78s at 85-in-90, 378s at 80-in-90 | what the gap leaves after the worst observed setup and upload | arithmetic on the three rows above |
| longest single command 593.1s | the granularity the soft-budget window has to exceed, since elapsed advances in whole commands | `chelis-cli::runtime_extent_claim_preparation` in the longest observed expansion shard, the command the deadline cut |
| batching discount 3.87x | why the balancing weight overstates a shard that runs | same shard: 64 targets carrying 1134s of weight executed in 293s of list and run time |
| 54% unmeasured, 30s default against a 28.4s p90 | why the projection is biased high a second way | 255 of 473 expansion targets have no row in `.config/ci-change-owned-durations.json`, and the default exceeds the p90 of the 218 that do |
| longest shard 1580s median, 2993s p90, 4410s max | the distribution the deadline is sized against | weight-fraction projection over the 38 dispatches under the current planner, from their `integration-package-expansion-*` receipts |
| estimate 3.73x over | why the plan's per-shard estimate is not a predicted duration | 96 shards that finished, estimated against actual |
| cost 64 to 81 job-min median, 73 to 111 p90, 74 to 158 worst | what raising the deadline costs | same 38 dispatches: job durations plus the projected extra for each cut shard |

The expansion executor splits ordinary targets into bounded exact-package groups, then issues one list command and one run command per group. The grouping limit changes command boundaries only: every selected target remains required in the informational receipt. Manual-only targets and targets with exact test exclusions remain singleton commands so ignored-test mode and filters cannot affect siblings. An ordinary expansion target whose listing contains no applicable non-ignored test is recorded as inspected and not applicable; an entirely inapplicable group does not launch a guaranteed-empty nextest run. Required change-owned targets remain fail-closed when no active test applies. Listings and JUnit are decomposed back into exact `package::target::test` evidence; a missing applicable result leaves that target incomplete and the shard unsuccessful. The executor stops after a shared 80-minute build/list/test deadline and writes an unsuccessful receipt containing completed-group evidence and the complete selected-target list. It terminates the active command's process group and starts no further command. Unfinished coverage is reported as the informational summary's unrun count rather than as a finding. The job's own 90-minute limit is what actually binds, and the executor's deadline sits ten minutes under it so a cut shard still finalizes and uploads its partial receipt; a shard the job limit kills produces none at all, so the two numbers move together. Most of that gap is job setup, which runs before the executor's clock starts and takes a measured median of 118s and maximum of 211s, rather than finalization, which merges and digests even a 98 MB JUnit in under two seconds. The receipt still records `soft_budget_exceeded` as a measurement, and nothing reads it as a verdict. Every nonempty shard runs the workspace-product build before target listing or execution. Test harnesses that link the uninstrumented runtime stage the runtime their own build carries through `chelis-runtime-bundle` ([spec/08 §2.1](../spec/08-backends.md)), so none of them reads a runtime from the target directory or the environment. Setup delays, external cancellation or runner loss can still prevent a receipt. Required change-owned execution remains fail-closed on incomplete coverage; its total job timeout includes the cold setup headroom described above.

Use `gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=runtime-representation` to validate that oracle alone, or `-f validation_scope=all` for the full run. For dtype Phase 0–3 alone, use `gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=dtype-phase3`. For runtime extent alone, use `gh workflow run heavy-e2e.yml --ref BRANCH -f validation_scope=runtime-extent`. A scoped result is only that oracle's verdict; the full-workspace shards, telemetry and main-only nightly report do not run. `gh workflow run macos-nightly.yml --ref BRANCH` retains its full selection. The exhaustive runtime-representation oracle has a 145-minute total job timeout, preserving mainline's 120-minute budget plus 25 minutes for cold Devenv setup. Dtype and runtime extent have 115-minute total job timeouts, preserving their 90-minute budgets; full Linux execution shards, script integrations and generalization shards have 85-minute total job timeouts, preserving their 60-minute budgets; the remaining extended Linux execution workers have 70-minute total job timeouts, preserving their 45-minute budgets. These increases do not change script execution deadlines. Darwin SMT remains native with a 60-minute job timeout. Exact-main evidence showed dtype finishing green after 80 minutes, while runtime representation completed Phase 1 and reached its final Python/DLPack cohort before the still-running binding census was cancelled at 90 minutes. These budgets preserve complete execution rather than converting slow success into cancellation; census work stays out of the other Linux workers. Existing ignored/manual gates still require their documented prerequisite and explicit invocation. The stdlib self-test corpus remains explicitly invoked nightly. Existing nightly failures must be recorded against a baseline, never treated as passing evidence.

The developer's `gate.py --fast`, `--validation`, `integration`, and full/manual commands retain their previous selections. The separate `gate.py ci-fast` stage owns the fixed standing hosted selection on pull requests and `main`. The required change-owned lane runs only for PR candidates and trusted exact-candidate dispatches; package expansion runs only through its explicit PR dispatch. A title or description edit reruns the dedicated acknowledgement check and changelog policy without entering compiler or Hull workflows, so the unchanged candidate's implementation contexts are neither cancelled nor replaced. A base retarget creates a required pending head receipt, validates the open PR's exact head/base and the checked-out two-parent merge, dispatches CI and Hull from the trusted new base, and closes the receipt only after both runs succeed. Run the owning phase oracle on the candidate when claiming phase completion.

### Measured figures for the changed-path classification stage

Measured on an Apple-silicon workstation. A figure here is a measurement, not
a constant: re-run the command before relying on one, and correct this table
rather than the prose that cites it.

| figure | conditions | how it was taken |
|---|---|---|
| 0.11-0.13s wall | any realistic change set, `--from-git` | `/usr/bin/time -p .venv/bin/python scripts/ci_change_owned.py classify-paths --from-git`, five runs. One path and fifty paths measure the same end to end in the argv form, which is the more size-sensitive of the two and showed no growth either |
| about 13 microseconds per path | the only part that grows with the change set | classification alone, in process, with config and packages preloaded: 0.1 ms at 8 paths, 6.7 ms at 500, 57.6 ms at all 4358, against a fixed floor of roughly 50 ms interpreter plus 30 ms `cargo metadata` |
| 0.05-0.06s wall | empty change set, which returns before `cargo metadata` | same command on an unchanged tree |
| `cargo metadata --no-deps --locked` 0.02-0.04s | warm, repeated invocation | the stage's dominant cost; 0.24s with dependencies, which it does not ask for |

The argv form is the one that grows visibly, and the shipped form does not
pay it: passing all 4358 paths as arguments costs about 0.36s wall, of which
only ~0.058s is classification and the rest is marshalling that many
arguments through `exec`. `--from-git` passes none.

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

## Rust build caches

Hosted jobs that compile Rust restore a `Swatinem/rust-cache` entry. Each entry belongs to one family named by `shared-key`, under the prefix `rust-v1-<runner environment>`. Its key also covers the runner OS, the installed rustc versions, the values of environment variables whose names begin with `CARGO`, `CC`, `CFLAGS`, `CXX`, `CMAKE` or `RUST`, and the Cargo manifests and lockfile. rust-cache keeps dependency artifacts and drops the workspace crates before saving, so an entry stays useful until a dependency, the toolchain or the job environment changes. A pull request whose dependencies differ from `main` restores `main`'s newest entry for the family and rebuilds what changed.

Each family has exactly one writer job. It saves only from `refs/heads/main`, and only after a successful build: rust-cache never re-saves a key it restored exactly, so a failed run's partial entry would otherwise stay live until a key input changed. A cache saved from another ref is visible only to that ref, so saving it would spend the shared budget on an entry no other run can restore.

Only the workflows listed as writers in `scripts/test_ci_cache_policy.py` may declare a Rust build cache save (`Swatinem/rust-cache`) or an `actions/cache` save. Each must show from its own definition that it saves from `main`'s code: it runs only on a push to `main`, on schedule or on dispatch, and checks out the triggering commit, never a ref taken from inputs, pull requests or the event payload. Every other workflow declares no such save or holds `cache-mode: read`. The rule does not cover the interpreter cache `astral-sh/setup-uv` saves by default or a cache saved inside a composite action.

- `ci-cache-warm.yml` writes the families that CI, Hull and package expansion restore: `lint-rust`, `linux-workspace`, `docs`, `smt-smt-build`, `smt-glibc231`, `backend-sanitizers`, `conformance`, `python-wheel-smoke` and `diagnostic-kind-oracle`. Its jobs mirror their consumers' environment, setup and cache inputs, so the key they save is the key the consumers compute and their build scripts see the same compilers and flags, and they run their consumers' own commands, tests included. It runs on a push to `main` that changes a key input (a Cargo manifest, the lockfile, the toolchain pin, Cargo's configuration, the workflow or the hosted setup action), weekly, and on manual dispatch. It has no pull-request trigger, and every job requires `main`.
- The scheduled workflows that build a family no pull request uses write it themselves from `main`: the macOS workspace and SMT lanes, the ecosystem drift builds, and the Hull nightly.
- Hosted runs of `heavy-e2e.yml` and `smt-full-prove.yml` restore the CI families and write none.
- Release builds keep no cache: a tag's cache is never restored by another run, and writing three release families from `main` would cost more than the rare release build saves.
- A job that can run on the self-hosted pool skips the GitHub caches there. The pool has its Kache compiler cache, and its rust-cache keys embed per-runner paths that never match another job.

`scripts/test_ci_cache_policy.py` locks these rules: the prefix and family on every cache step, one main-only writer per family that saves only on success, the closed list of writing workflows and their main-only proof, the same declared key inputs and build environment across a family's jobs, and a cache writer that runs only commands its consumers run. It reads what the workflows declare; a variable a step exports through `GITHUB_ENV` is outside it.

To start every family afresh, for example after a hosted system-package change that Cargo cannot see, raise the `rust-v1` prefix in every cache step in one change.

The families hold roughly 6 to 7 GB of live entries: about 4 GB that pull requests restore and 2.5 GB of nightly families. A dependency change leaves the previous generation in place until GitHub evicts it, and GitHub deletes any entry unused for seven days. The repository's Actions cache limit (Settings > Actions > General) should be at least 20 GB, so that once-a-day nightly entries and a full stale generation fit without the least recently used eviction reaching a live family. `cache-prune.yml` deletes closed pull-request entries and duplicate `main` generations weekly.

## Python execution ownership and timing

`python scripts/ci_script_tests.py pr` runs the cheap discovered tests; `nightly`
runs the compiler-dependent classes listed in that script. The selection receipt
assigns every discovered test to exactly one of PR, script nightly, census or
profile-oracle ownership. Classes absent from discovery and absent census control
identities fail selection. New methods inherit their class's cadence; ordinary
new classes enter the PR selection, whose job has a 35-minute total timeout: its
prior ten-minute budget plus 25 minutes for cold Devenv setup.

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

When a change-owned or manual package-expansion shard selects a census, its
uploaded shard receipt also contains `census-timings/*.jsonl` for the nested
commands. The directory is inside the receipt so a failed or partial shard
can upload the timings it reached; no prior timing or build result satisfies a
test obligation.

The same diagnostic JSONL now names each selected runtime-extent target and
each case in `claimed_extent_contract`, plus individual supervised capacity
controls and the wire/binding verification stages. The runtime-extent job
uploads `runtime-extent-timings` for full scheduled or selected manual runs;
dtype and PR shard timing artifacts retain the capacity rows they execute.
Each row has a kind and name, a start event, and a finish event with elapsed
seconds and outcome. A start without a finish identifies interrupted work.
Stages and subprocesses can overlap, so their seconds cannot be added to
obtain job wall time. These rows are diagnostics outside the authoritative
selection, evidence digests and pass/fail decisions.
