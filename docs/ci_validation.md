# CI validation cadence

For the actions to take when preparing a PR, read
[Changing tests, inventories and protected contracts](guard_changes_for_pr_authors.md).

Ordinary PRs and main pushes use Linux. Passing required PR checks is **not a phase acceptance result** for a full or feature-specific oracle that runs nightly.

| Owner | Cadence | Coverage |
|---|---|---|
| `ci.yml` `ci-fast` | PR and main push, with the existing docs-only skip | Every default-feature library/binary unit target and the reviewed `standing_target` identities in `.config/ci-test-targets.toml`; 20-minute limit |
| `ci.yml` change-owned shards and report | PR and main push, with the existing docs-only skip | Every default-enabled integration target added or directly modified by the change, or its exact reviewed alternative owner; four deterministic shards with a 20-minute limit each |
| `ci.yml` package-expansion shards and summary | After the required change-owned report on PR and main push | Other default-enabled integration targets in directly selected packages, excluding exact reviewed target/test rows; four informational shards with a 20-minute hard limit and a separate non-required summary |
| `ci.yml` retained workers | PR and main push | Rust policy and doctests, Python/script units selected by `ci_script_tests.py pr`, focused SMT plus its existing Deep-obligation integration target, Linux glibc compatibility, Docs, backend sanitizer units and explicit backend doctests; change-triggered diagnostic mutation and rejection liveness |
| `conformance.yml` | PR and main push | Existing frozen Hull conformance gate |
| `heavy-e2e.yml` | Daily 03:17 UTC and manual dispatch | Full non-ignored default workspace across four workspace shards plus the dtype owner, script integrations, exhaustive generalization feature partitions, dtype Phases 0–3, faithful observation Phase 2, ownership Phase 2 and launch, runtime representation Phase 0, frontend/domain support, and full backend sanitizer integration coverage |
| `macos-nightly.yml` | Daily 04:17 UTC and manual dispatch | Both Mac workspace partitions, both Clippy configurations, architecture/ABI/Metal smokes, and Darwin SMT |
| `build-cvc5.yml` Darwin producer | Daily 01:17 UTC and manual dispatch | Missing Darwin prebuilt assets; relevant main pushes may produce Linux assets only |
| `smt-full-prove.yml` | Existing nightly/manual cadence | Full SMT proof validation |
| `release.yml` and release-triggered Nix checks | Release/manual cadence | Existing shipping-artifact validation, including Mac |

The Linux workspace worker passes `--ignore-default-filter` and deliberately includes the PR selection. Only `chelis-compiler-api::capacity_census_wire` and `chelis-python::capacity_census_bindings` are excluded: the dtype worker executes both. The generalization worker uses the same census exclusions; census authority is checked on the default-feature configuration. Executable listing set-math proves that workspace plus dtype still covers every non-ignored test in the unfiltered corpus, with none in neither selection and no census test in both. The two selections do overlap elsewhere: the dtype oracle also owns several non-census binaries.

The regular PR planner compares Cargo metadata at the change base and candidate. On a pull request it requires the checked-out synthetic merge commit to have exactly two parents and requires the event head to be `HEAD^2`; on a main push it uses the event's before/after commits. Its rename-aware NUL-delimited diff assigns every changed path one final disposition. Package roots map through Cargo metadata, reviewed shared paths map through `path_rule` rows, and unknown, stale, duplicate, or ambiguous mappings fail planning. Added targets and targets whose exact candidate `src_path` changed enter the required change-owned set. Other eligible targets in selected packages enter the disjoint informational package-expansion set.

`.config/ci-test-targets.toml` is the versioned ownership manifest for this surface. `standing_target` rows feed `ci-fast`; `target_exclusion` and `test_exclusion` rows name their exact alternative workflow, job, cadence, reason, and tracking issue; and `path_rule` rows assign shared paths to exact packages or an existing automated owner. Other prose paths use the existing docs-only classifier, including its executable-document exceptions; a new changelog fragment needs no manifest row. Package qualification is retained throughout, including execution, so equal target names in different packages cannot create a Cargo selector cross product.

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

`scripts/ci_test_targets.py` first builds workspace product libraries and binaries so cross-package `cargo_bin` users and generated-C tests have their CLI and runtime static library. It lists all units and globally unique standing test names together, then lists shared names with exact package selectors. The combined binary receipt must contain every unit (including zero-test unit binaries) and exactly the selected integrations before any group executes. Each group runs once without default filters; command, listing, timing, and JUnit receipts preserve the whole union. Missing unit binaries, duplicate or unexpected integration binaries, empty selected integrations, filtered non-ignored tests, or missing JUnit fail the worker. Ignored tests remain ignored; this pass neither deletes tests nor changes language semantics.

`Integration Tests (Linux)` fails closed on both the standing fast worker and the required change-owned report. The four required shards are assigned by `sha256(package + "::" + target) mod 4`; their report rejects missing shards, digest disagreement, duplicate execution, uncovered selected targets, executed exclusions, and test failure. The `integration-change-plan`, per-shard receipts, required report, JUnit, commands, selected/executed lists, and timings are retained for 14 days.

Each worker downloads the current run's plan after Cargo cache restoration and before execution. The cache may replace `target/`, so it cannot own the plan; a missing artifact remains a job failure.

Package expansion uses a separate four-shard worker pool and starts only after the required change-owned report succeeds. Its failures, missing shards, exclusions, and timing-budget overruns are recorded by `Informational Package Expansion Summary`, but neither that summary nor its workers feed `Integration Tests (Linux)`. This sequencing keeps the trial from cancelling, starving, or changing the required verdict. Nightly JUnit reports stay in their producing workflow. The Linux nightly report inspects every execution worker and opens a failure tracker on non-success; a manual branch run cannot close a main-nightly tracker.

Use `gh workflow run heavy-e2e.yml --ref BRANCH` or `gh workflow run macos-nightly.yml --ref BRANCH` for candidate validation. Full Linux execution shards, dtype, script integrations and generalization shards have 60-minute timeouts; other extended Linux workers have 45 minutes, and Darwin SMT has 60. Dtype previously exhausted 45 minutes; its 60-minute allowance preserves complete execution while census work is removed from the other Linux workers. Existing ignored/manual gates still require their documented prerequisite and explicit invocation. The stdlib self-test corpus remains explicitly invoked nightly. Existing nightly failures must be recorded against a baseline, never treated as passing evidence.

The developer's `gate.py --fast`, `--local`, `integration`, and full/manual commands retain their previous selections. The separate `gate.py ci-fast` stage owns the standing hosted selection; the change-owned and package-expansion lanes exist only in hosted PR/main CI. Run the owning phase oracle on the candidate when claiming phase completion.

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
