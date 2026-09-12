# CI diet regression analysis, 2026-09-10 / 2026-09-11

**Question asked:** which issues filed in the ~19 hours after `062c29c19` were caused by
pull requests merging to `main` red against tests that had just been moved off the
per-pull-request path, and which pull requests were responsible.

**Scope of evidence:** the repository at `origin/main` `d861a6c6f`, the GitHub Actions API,
issues #1738-#1808, and four local reproduction runs in throwaway detached worktrees
(removed and pruned afterwards). Nothing was written in the primary checkout.

---

## 1. Headline

Four findings, in order of how much they explain. The fourth was measured after the rest and corrects them.

1. **The commit in the original question is the smallest of four.** `062c29c19` (#1736) is the
   tail of a four-pull-request CI diet Jeff merged on 2026-09-10. The two that removed the test
   mass are **#1723** and **#1726**, merged 13:46 and 14:22 UTC. #1736 merged at 18:04 UTC and
   moved 47 compiled Python methods; every one of them passed in the nightly.

2. **One pull request explains most of the breakage: #1693.** `22b193cf7`, *"Preserve computed
   reshape claims and broadcast unit checks through calls"*, merged 18:18 UTC on 2026-09-10,
   four hours after the coverage was removed, as the first substantive compiler change under the
   new regime. It is the first bad commit for **11 of the 13** failures in the 2026-09-11
   Linux Extended Validation nightly, and for **all ten** `chelis-cli` failures the #1776 audit
   found, AND for the 170x evaluation regression in chelis#1829 (section 12.5). It merged with
   21 required contexts and no workspace suite.

3. **The nightly backstop is not actually running.** The `Full Workspace (Linux)` job was
   cancelled at its 60-minute timeout having executed **1,714 of 10,365 tests (16.5%)**. Every
   "runs only in the unfiltered nightly" sentence in #1775/#1776/#1777/#1778/#1779 is, in
   practice, "runs nowhere."

4. **The nightly's 60-minute budget is not the binding constraint; four tests are.** Sharding
   `full-workspace` four ways left one shard finishing in 25 minutes and three blocked past 45
   minutes each on `csv_io`, `json_io` and `issue_1314_json_bigint` rows that took 6 to 14 seconds
   a day earlier. That is a 170x to 207x regression, chelis#1829. **Section 12 is an addendum
   written after the rest and corrects sections 1, 4, 7 and 9.5 on this point.**

---

## 2. What the four CI pull requests actually changed

| PR | commit | merged (UTC) | what left the per-PR path |
|---|---|---|---|
| #1721 | `3715225ee` | 2026-09-10 13:15 | `macOS Smoke`, `macOS Workspace (shard 1/2, 2/2)` |
| **#1723** | `0c40a3203` | 2026-09-10 13:46 | `Dtype Phase 0-3 Oracle`, `Faithful Observation Phase 2 Oracle`, `Compiled Value Ownership Phase 2 Oracle`, `Runtime Representation Phase 0 Oracle`, `Typecheck Level Generalization Oracle` + its 4 shards, `ProfilePartitionTests`, the frontend/domain integration support subsets |
| **#1726** | `d1363ce57` | 2026-09-10 14:22 | `Workspace Tests (Linux, shard 1/2, 2/2)` replaced by `Fast Tests (Linux)` = default-feature lib/bin units plus the 26 targets named in `.config/ci-test-targets.toml` |
| #1736 | `062c29c19` | 2026-09-10 18:04 | 47 compiled Python test methods to a new nightly `Script Integration Tests (Linux)` job, 36 census methods to the dtype oracle job; also excluded two census nextest binaries from the nightly workspace and generalization runs |

### Measured before and after

| | before (PR #1720, merged 13:14 UTC) | after (PR #1799, merged 2026-09-11 12:13 UTC) |
|---|---|---|
| required check contexts | 34 | 19 |
| tests executed per PR | 4,787 + 4,789 = **9,576** (two shards, ~16 min each) | **4,098** in 174 s |
| whole-workspace nets | 2 (`Workspace Tests` and the generalization oracle) | 0 |
| phase oracles on PR | 5 | 0 |
| macOS | 3 jobs | 0 |

`Integration Tests (Linux)` survives as a required context but now executes no tests at all. Its
only step is `python3 scripts/ci_require_success.py ci-fast=...`, which re-reports the fast
worker's result.

### The two nets that were removed

Both #1723 and #1726 removed a *whole-workspace* net, 36 minutes apart:

- #1726 removed `gate.py integration --tests-only --partition hash:N/2`, which ran
  `cargo nextest run --workspace` under the `ci` profile.
- #1723 removed the 4-shard
  `cargo nextest run --workspace --profile ci-full --ignore-default-filter --features chelis-types/generalize-sweep-oracle`,
  which excluded exactly two things: `chelis-cli::stdlib_typecheck_cache_concurrency` and one
  `issue_1293_redteam_round4` test.

A regression escapes only when **both** are gone, so the two are jointly responsible for the
overlapping failures. If you ask which single restore would recover the most, the answer is
**#1723**: the generalization oracle covers everything the workspace suite covered, and it is
additionally the *only* net that covered the feature-flagged failures (#1746) and the
`Dtype Phase 0-3 Oracle` digest regression.

### Ranking the coverage removals by damage

| rank | PR | issues it let through | sole net for |
|---|---|---|---|
| 1 | **#1723** | 11 filed + 1 unfiled | #1746; the dtype frozen-digest regression; #1781's class |
| 2 | **#1726** | 10 filed | (none on its own; the generalization oracle duplicated it) |
| 3 | #1736 | 0 let through, 1 *caused* (#1781) | n/a |
| 4 | #1721 | none identified | n/a |

---

## 3. Ranking the code pull requests by breaks caused

| rank | PR | commit | merged (UTC) | failures | issues |
|---|---|---|---|---|---|
| **1** | **#1693** *Preserve computed reshape claims and broadcast unit checks through calls* | `22b193cf7` | 09-10 18:18 | **11 tests** | #1746, #1747, #1775, #1776, #1779 |
| 2 | #1749 + #1751 + #1758 (jointly) | `133f510bd`, `324128145`, `e813415d0` | 09-10 19:31 / 20:00 / 21:12 | 1 oracle (`Dtype Phase 0-3`) | **none filed** |
| 3 | #1799 *fix(c): isolate Random state across public invocations* | `d861a6c6f` | 09-11 12:13 | 2 tests + the phase-b oracle | #1808 |
| 4 | #1750 *fix(eval): restore scalar gradient result carriers* | `4c2854b97` | 09-10 19:37 | 1 test | #1777 |
| 5 | #1770 interacting with #1307 | `fcbc7e96a`, `3dc3f54f6` | 09-11 04:51 / 04:26 | 1 test | #1787 |
| 6 | #1773 *Phase B2b-1: named result claims survive the call boundary* | `fd6fc6f5d` | 09-11 07:06 | 1 test | #1789 |
| 7 | #1736 (the CI pull request itself) | `062c29c19` | 09-10 18:04 | 1 test (its own new assertion) | #1781 |
| - | #1547 / #1590 (older, surfaced not caused) | | | 1 test | #1778 |

### 3.1 #1693 in detail

Measured locally: two detached worktrees at `062c29c19` (#1693's parent) and `22b193cf7`
(#1693), `cargo nextest run -p chelis-cli`, plus a second pass under
`--features chelis-types/generalize-sweep-oracle`.

| test | at `062c29c19` | at `22b193cf7` | tracked in |
|---|---|---|---|
| `chelis-cli::cli::build_c_with_seed_uniform_like_succeeds` | PASS | FAIL | #1776 |
| `chelis-cli::cli::build_hip_emits_pad_kernel` | PASS | FAIL | #1776 |
| `chelis-cli::cli::build_hip_emits_shrink_kernel` | PASS | FAIL | #1776 |
| `chelis-cli::issue_1222_root_alias_ownership::conditional_over_existing_bindings_claims_no_third_allocation` | PASS | FAIL | #1776 |
| `chelis-cli::issue_1222_root_alias_ownership::seeded_block_returning_an_outer_binding_claims_no_second_allocation` | PASS | FAIL | #1776 |
| `chelis-cli::redteam_731_adversarial::redteam_baked_seed_deterministic_and_cross_lane` | PASS | FAIL | #1776 |
| `chelis-cli::issue_1564_bounded_tensor_cast::result_constraints_actualize_bounded_tensor_targets` | PASS | FAIL | #1746 |
| `chelis-cli::issue_1564_bounded_tensor_cast::result_only_cast_instances_do_not_share_precision` | PASS | FAIL | #1746 |
| `chelis-cli::issue_368_grad_concat_windows::issue_368_runtime_symbolic_window_grad_is_half_everywhere` | PASS | FAIL | #1775 (reporter's `git bisect`) |
| `chelis-cli::reshape_symbolic_dim_vmap_column::vmap_over_symbolic_column_builds_without_symbolic_dim_ice` | - | FAIL | #1779 (inferred, not bisected) |
| `chelis-compiler-api::wire_dag_vocabulary::wire_dag_operation_vocabulary_is_pinned_to_version_9` | - | FAIL | #1747 |

Two corrections to the filed issues follow from these runs:

- **#1746's framing is wrong.** It reads as though #1724's bounded-cast fix shipped incomplete
  under the generalization feature. It did not: both rows **pass** at #1724's own merge commit
  `3c0d46137` with the feature enabled, and at `062c29c19`. They fail at `22b193cf7`. The cause
  is #1693.
- **#1747 is #1693 too.** #1693 bumped `WIRE_DAG_SCHEMA_VERSION` from 9 to 10 in
  `crates/chelis-compiler-api/src/schema.rs:2011`. The guard that pins the vocabulary to 9 landed
  in #1716 (`695eb6840`, 14:59 UTC), 37 minutes after #1726 removed the job that would have run it.

`22b193cf7` touched `chelis-backend-c/src/emit.rs`, `chelis-ir/src/dag.rs`, `eval.rs`, `grad.rs`,
`host.rs` (+379 lines) and added `chelis-ir/src/host/staged.rs` (676 new lines) across roughly 30
commits. It is exactly the shape of change the removed whole-workspace nets existed for.

### 3.2 The one regression nobody has filed

The `Dtype Phase 0-3 Oracle` failed in the nightly with:

```
DTYPE PHASE 3 ORACLE: FAIL: crates/chelis-cli/tests/parity.rs test parity_corpus_is_complete
changed its frozen Phase 3 definition
(expected 6603a110d51e7c5629d9d6e1026f2ca6ecfa4e05d565aa2df32caa91180e41ee,
 got     15112c243062ff87657d9ac7bfd2fd16b680c5e397f6bea8e4003a7cb106e03c)
```

The digest lives at `scripts/faithful_observation_phase3_oracle.py:185`, reached through
`dtype_phase3_oracle.py`'s `faithful.preflight_violations()`. #1693 recorded that value. Then
**#1749**, **#1751** and **#1758** each added an example to the frozen `parity_corpus_is_complete`
list without updating it. That oracle was a required per-pull-request context until #1723 moved it
at 13:46 UTC, so all three would have been caught. No issue exists for this; only the
auto-opened #1783 covers it indirectly.

### 3.3 A pattern worth naming

Every one of the seven pull requests Jeff merged that evening added **its own** new test target to
`.config/ci-test-targets.toml`:

| PR | target it added |
|---|---|
| #1749 | `chelis-cli::issue_1710_c_source_names` |
| #1750 | `chelis-compiler-api::issue_1741_scalar_gradient_api`, `chelis-cli::issue_1741_scalar_gradient_cli` |
| #1751 | `chelis-cli::issue_1732_integer_host_literals` |
| #1756 | `chelis-compiler-api::issue_1650_empty_tensor_api`, `chelis-cli::issue_1650_empty_tensor_cli` |
| #1757 | `chelis-compiler-api::issue_1738_transcript_capture` |
| #1759 | `chelis-cli::issue_1418_recursive_cast_targets` |
| #1758 | `chelis-cli::issue_1640_value_root_actualization` |

New behaviour is defended; the older suites that the same behaviour change breaks are no longer
compiled. **#1750 is the clean illustration**: it added two acceptance targets for issue #1741 and
in the same change broke `chelis-compiler-api::issue_257_tensor_scan_host_runtime`, which asserts
`ExecutionValue::Tensor` on a scalar cotangent and is not in the selection (#1777).

---

## 4. The Linux Extended Validation nightly

### 4.1 The run

| | |
|---|---|
| run | [34558777344](https://github.com/Chelis-Lang/chelis/actions/runs/34558777344) |
| trigger | `schedule`, 2026-09-11 03:31:41 UTC |
| head | `e813415d09fced6ac6c22813dc717bb3c3e2f02f` (#1758) |
| conclusion | failure |
| tracking issue | #1783, auto-opened by `Surface nightly status`, auto-closes on the next pass |

| job | result |
|---|---|
| Compiled Value Ownership Phase 2 Oracle | success |
| Faithful Observation Phase 2 Oracle | success |
| **Script Integration Tests (Linux)** | **success** (the job #1736 created) |
| **Dtype Phase 0-3 Oracle** | **failure** |
| **Full Workspace (Linux)** | **cancelled at the 60-minute timeout** |
| Runtime Representation Phase 1 Oracle | success |
| Backend Sanitizers (Full) | success |
| **Typecheck Level Generalization Oracle, shards 1-4 + aggregate** | **failure** |
| Integration Support (Linux, frontend / domain) | success |
| Surface nightly status | success (opened #1783) |
| Extended Test Telemetry | skipped (gated on the three red jobs) |

### 4.2 Full Workspace (Linux): the backstop did not finish

Started 03:31:44, cancelled 04:36:45. `timeout-minutes: 60`. It reached test **1,714 of 10,365**
before the runner received the shutdown signal, then failed its artifact upload
(`No files were found with the provided path: target/nextest/ci-full/junit.xml`) and its
`chelis_std_self_test_corpus` step exited 143.

This is the most consequential line in the whole analysis. #1726's premise is that the de-selected
tests are covered nightly. At 16.5% completion in a cold-cache 60-minute box, they are not. The
two nightlies before this one (2026-09-09 `34328558655`, 2026-09-10 `34454619309`) also failed, on
the older single `Heavy E2E (Linux)` job; 2026-09-08 `34203981293` was the last green nightly.
That is why #1693's breakage sat undetected for a full day.

### 4.3 Dtype Phase 0-3 Oracle

Single failure, the frozen-digest drift described in §3.2. Caused by #1749, #1751 and #1758;
unfiled.

### 4.4 Typecheck Level Generalization Oracle: all four shards

Command per shard:
`cargo nextest run --workspace --profile ci-full --ignore-default-filter --features chelis-types/generalize-sweep-oracle --no-fail-fast -E 'not (...)' --partition hash:N/4`

| shard | job | tests run | failed | wall |
|---|---|---|---|---|
| 1/4 | 103136993210 | 2,614 | 2 | 2,489.5 s |
| 2/4 | 103136993004 | 2,629 | 6 | 2,552.1 s |
| 3/4 | 103136993038 | 2,560 | 2 | 915.1 s |
| 4/4 | 103136993107 | 2,559 | 3 | 1,942.4 s |
| | | **10,362** | **13** | |

The 13 failures, with attribution:

| # | shard | test | cause | issue |
|---|---|---|---|---|
| 1 | 1 | `chelis-cli::cli::build_hip_emits_pad_kernel` | #1693 | #1776 |
| 2 | 1 | `chelis-compiler-api::issue_1277_host_lane_routing::host_applied_mismatch_preserves_the_current_local_failure` | #1547/#1590 expand/insert split; test authored in #1664 against the pre-split spelling | #1778 (closed) |
| 3 | 2 | `chelis-cli::issue_1222_root_alias_ownership::conditional_over_existing_bindings_claims_no_third_allocation` | #1693 | #1776 |
| 4 | 2 | `chelis-cli::issue_1222_root_alias_ownership::seeded_block_returning_an_outer_binding_claims_no_second_allocation` | #1693 | #1776 |
| 5 | 2 | `chelis-cli::issue_1564_bounded_tensor_cast::result_only_cast_instances_do_not_share_precision` | #1693 | #1746 |
| 6 | 2 | `chelis-cli::issue_368_grad_concat_windows::issue_368_runtime_symbolic_window_grad_is_half_everywhere` | #1693 | #1775 (closed, fixed by #1780) |
| 7 | 2 | `chelis-cli::redteam_731_adversarial::redteam_baked_seed_deterministic_and_cross_lane` | #1693 | #1776 |
| 8 | 2 | `chelis-cli::reshape_symbolic_dim_vmap_column::vmap_over_symbolic_column_builds_without_symbolic_dim_ice` | #1693 | #1779 |
| 9 | 3 | `chelis-cli::issue_1564_bounded_tensor_cast::result_constraints_actualize_bounded_tensor_targets` | #1693 | #1746 |
| 10 | 3 | `chelis-compiler-api::wire_dag_vocabulary::wire_dag_operation_vocabulary_is_pinned_to_version_9` | #1693 (schema 9 to 10) | #1747 (closed) |
| 11 | 4 | `chelis-cli::cli::build_c_with_seed_uniform_like_succeeds` | #1693 | #1776 |
| 12 | 4 | `chelis-cli::cli::build_hip_emits_shrink_kernel` | #1693 | #1776 |
| 13 | 4 | `chelis-compiler-api::issue_257_tensor_scan_host_runtime::issue257_grad_unrelated_tensor_scan_def_does_not_block` | #1750 | #1777 |

**11 of 13 are #1693.**

Two clusters are visible in the failure text. The seed rows (1, 7, 11, 12 plus the HIP kernels)
expect `CHELIS_EFFECTIVE_UNIFORM_SEED(7ULL)` in the emitted C and the pad/shrink kernel sources;
the emitter still contains those spellings at `crates/chelis-backend-c/src/emit.rs:3487-3491`, so
the likely mechanism is #1693's staged host routing selecting a different lane rather than a
changed literal. The `issue_1222` rows count retains and releases that no longer balance
(`got 3 retain(s) and 5 release(s)`). Neither has been root-caused yet; only the first bad commit
is established.

### 4.5 Script Integration Tests (Linux): the job #1736 created

Passed. Its selection receipt reads:

```json
{"pr": 2482, "nightly": 47, "census": 36, "profiles": 12}
```

`Ran 47 tests in 595.588s`. So #1736 left 2,482 Python methods on the pull-request path, moved 47
to this job and 36 to the dtype oracle job. **None of them failed.** No filed issue traces to a
test #1736 moved off the pull-request path.

---

## 5. Issue-by-issue map

48 issues were filed between 2026-09-10 14:22 UTC and 2026-09-11 13:00 UTC. Eleven belong to this
class.

| issue | state | test or oracle that went red | caused by | hidden by |
|---|---|---|---|---|
| #1746 | open | `issue_1564_bounded_tensor_cast` x2 (generalization lane) | **#1693** | #1723 |
| #1747 | closed | `wire_dag_vocabulary::..._pinned_to_version_9` | **#1693** | #1726 + #1723 |
| #1775 | closed (#1780) | `issue_368_grad_concat_windows` | **#1693** | #1726 + #1723 |
| #1776 | open | nine `chelis-cli` tests | **#1693** (all nine) | #1726 + #1723 |
| #1777 | open | `issue_257_tensor_scan_host_runtime` | **#1750** | #1726 + #1723 |
| #1778 | closed | `issue_1277_host_lane_routing` | #1547/#1590, surfaced not caused | #1726 + #1723 |
| #1779 | open | `reshape_symbolic_dim_vmap_column` | **#1693** | #1726 + #1723 |
| #1781 | open | `ProfilePartitionTests::test_linux_workspace_and_dtype_cover_the_complete_census_partition` | **#1736 itself** | #1723 |
| #1787 | open | `issue_1739_diagonal_runtime_bound::no_shipped_example_gains_a_return_boundary_guard` | **#1770** x **#1307** | #1726 |
| #1789 | open | `chelis-types::expand_insert_source_literal_inventory` | **#1773** | #1726 |
| #1808 | open | `runtime_extent_slice_b` x2 + phase-b oracle | **#1799** | #1726 |
| #1783 | open | the nightly itself | see §4 | n/a |

### #1781 is a defect in #1736

#1736 added
`ProfilePartitionTests::test_linux_workspace_and_dtype_cover_the_complete_census_partition`, whose
final assertion is:

```python
self.assertEqual(active(_list_filterset(f"not ({CENSUS_SELECTOR})")), full - census)
```

The left side runs `cargo nextest list --workspace --ignore-default-filter -E ...`. The right side
derives from `cls.full = _list_profile(None)`, which honours the default profile's
`default-filter`. The 17 names in the failure are **exactly** the `[profile.default]
default-filter` exclusion set in `.config/nextest.toml`: `stdlib_typecheck_cache_oracle` (6) +
`stdlib_typecheck_cache_concurrency` (3) + `monomorphization_build` (2) + six named `cli`,
`bundled_chelis_std_loader`, `reef_install_from_monorepo` and `issue_1293_redteam_round4` tests.
The assertion cannot pass as written, and it landed into a class #1723 had made nightly-only
36 minutes earlier, so nothing ran it before merge.

---

## 6. What is *not* caused by the CI change

The other ~36 in-window issues are red-team semantic findings on dtype resolution, extent claims,
grad, and the C/HIP backends, produced by the review rounds running on the in-flight #1277 slices:
#1738-#1741, #1743-#1745, #1748, #1753-#1755, #1760-#1768, #1771, #1772, #1782, #1786, #1788,
#1791, #1794-#1798, #1800-#1805. They would have been filed regardless.

Two further `area:ci` issues in the window are also unrelated:

- **#1742** (closed by #1769): `runtime_extent_oracle.py` receipt drift from #1664 and #1668. That
  oracle was never a pull-request check, so the diet did not hide it.
- **#1784**: the `tests/io/csv.ch` rows sit about 3 s under the harness's 30 s per-test cap and
  time out under load. A boundary-number flake of the #1607 shape, independent of the diet.

---

## 7. Suggested repairs, in priority order

1. **Make the nightly finish.** `Full Workspace (Linux)` covering 16.5% of its suite is not a
   backstop. Either shard it the way the generalization oracle is sharded, raise the timeout well
   past 60 minutes, or warm its cache. Until it completes, #1726's premise does not hold.
2. **Restore one whole-workspace net to the pull-request path**, or accept that per-pull-request
   green means "units plus 26 targets". The generalization oracle is the higher-value one to
   restore first: it subsumes the workspace suite and is the only net for the feature-flagged lane.
3. **Root-cause and repair #1693's 11 failures.** #1775 is already fixed by #1780; the remaining
   ten are open across #1746, #1747 (closed but verify), #1776 and #1779.
4. **File and fix the `Dtype Phase 0-3 Oracle` digest regression** (§3.2). It is the only red row
   in the nightly with no issue behind it.
5. **Fix #1781's assertion** so it compares like listings, since it can never pass today.
6. **Reconsider the "add my own target" pattern** (§3.3). A pull request that changes shared
   lowering behaviour needs the old suites compiled, not just a new one added.

---

## 8. Merge volume and exposure

**Window:** `062c29c19` (2026-09-10 18:04 UTC) to `2c4af13f9` (2026-09-11 13:53 UTC), **18.8 hours**.

| | |
|---|---|
| pull requests merged in the window | **28** (1.49 per hour) |
| merged since #1726, the selection cut (14:22 UTC 09-10) | **38** |
| of the 28, how many touched `crates/`, `packages/` or `examples/` | **28 of 28**; none was docs-only |
| required check contexts each one merged under | **19** (21 for the two carrying openspec legs), down from 34 |

### 8.1 How many merged with a failing test they could not see

Two honest readings, because the phrase can mean "introduced one" or "merged while one existed".

**Reading A, introduced a break invisible to its own CI: at least 10 of 28, 36%.**

| PR | commit | merged UTC | what it turned red | contexts |
|---|---|---|---|---|
| #1693 | `22b193cf7` | 09-10 18:18 | 11 tests | 21 |
| #1749 | `133f510bd` | 09-10 19:31 | `Dtype Phase 0-3 Oracle` frozen digest (first to break it) | 19 |
| #1750 | `4c2854b97` | 09-10 19:37 | `issue_257_tensor_scan_host_runtime` | 19 |
| #1751 | `324128145` | 09-10 20:00 | same digest, edited the frozen definition again | 19 |
| #1759 | `9a1ec0ae8` | 09-10 20:25 | same digest, added `recursive_cast_targets.ch` to the frozen list | 19 |
| #1758 | `e813415d0` | 09-10 21:12 | same digest, edited it a fourth time | 19 |
| #1733 | `8ae55787f` | 09-11 10:01 | **two breaks**: shipped `examples/dropout_fixed_stream.ch`, which the shipped-example census builds to C and which cannot build (#1192 unimplemented, #1787); and added two `#[test]` functions to `capacity_census_tripwire.rs` without re-freezing the runtime-representation Phase 1 selection, 80 to 82 (#1817) | 19 |
| #1770 | `fcbc7e96a` | 09-11 04:51 | `no_shipped_example_gains_a_return_boundary_guard` | 19 |
| #1773 | `fd6fc6f5d` | 09-11 07:06 | `expand_insert_source_literal_inventory` | 19 |
| #1799 | `d861a6c6f` | 09-11 12:13 | `runtime_extent_slice_b` x2 and the phase-b oracle | 19 |

**This is a floor, not an audit.** Every entry was found by a person happening to run a full suite
while validating unrelated work, never by a check. Establishing the true number requires running
the whole workspace at each of the 28 merge commits, which nothing currently does. #1307 is
excluded: it added the 32nd example, but nothing was red at its merge; the break materialised when
#1770 hard-coded 31 twenty-five minutes later.

**Reading B, merged onto a tree that already had invisible failing tests: 27 of 28, 96%.**

#1693 is merge number 1 in the window. From `22b193cf7` onward `main` has carried at least eleven
red tests continuously, and no pull-request check could see any of them. Every one of the following
27 merges therefore showed green against a tree that was not. (Arguably 28 of 28: #1778's
`host_applied_mismatch_preserves_the_current_local_failure` was already red before the window
opened.)

### 8.2 The repair loop is now a measurable share of throughput

At least 5 of the 28 merges are repairs of issues filed inside this same window:

| repair PR | closes |
|---|---|
| #1769 `0e858f184` | #1742 |
| #1780 `a80e2ac41` | #1775 |
| #1807 `48ee57c3a` | #1764 (still open) |
| #1810 `6be0e7d59` | #1808 |
| #1809 `2c4af13f9` | #1803 |

That is **18% of merge traffic** already spent unwinding findings from the preceding day, before
the eleven #1693 failures, the dtype digest, and #1781 have been touched at all.

---

## 9. Repair-cost triage: what is a regenerated hash and what is a real break

The twelve items in the CI-regression class do not cost the same. Classified by what the repair
actually is, with the evidence for each classification.

### 9.1 Regenerate, re-pin, or re-freeze. No behaviour question at all. (5)

This is the single largest class, and every member has the same shape: a hand-maintained frozen
list, digest, or count that a legitimate merge invalidated, with no per-pull-request check to say
the update was owed.

| item | the repair | status |
|---|---|---|
| #1818 | one line: the frozen digest at `scripts/faithful_observation_phase3_oracle.py:185`. Four merges edited `parity_corpus_is_complete` without moving it (#1749, #1751, #1759, #1758); every edit was a legitimate example addition. | open, filed 09-11 14:30 |
| #1817 | re-freeze `spec/design/runtime_representation_phase1_tests.json`. #1733 added two `#[test]` functions to `capacity_census_tripwire.rs`, 80 to 82, and did not move the frozen selection. | open, filed 09-11 14:26 |
| #1747 | re-pin the WireDag vocabulary guard to schema 10 and the three operations version 10 added (`checked_reshape_extent`, `checked_unit_axis`, `mod`). The schema bump in #1693 was legitimate. | closed by #1307 |
| #1789 | the inventory row for `axis_sources.rs` says 2 `expand` literals where there are 3. The count is one line; attached to it is one genuine judgement call, whether an `insert` counterpart is owed. | open |
| #1787, first half | `EXECUTABLE_PHASE_0_EXAMPLES` is pinned to 31; `examples/` now holds 36. | open |

Adjacent, and the same class though not caused by the diet: **#1742**, whose hand-maintained
receipt tuples in `runtime_extent_oracle.py` drifted after #1664 and #1668 renamed and added tests.
#1588 was the first instance. That makes **six** instances of one defect class inside four days,
which is the argument for deriving these artifacts from their sources rather than repairing each
witness.

### 9.2 A test asserted an incidental spelling. The behaviour was correct. (4)

| item | the repair | status |
|---|---|---|
| #1781 | the new assertion compares `_list_filterset(...)` (which passes `--ignore-default-filter`) against `_list_profile(None)` (which honours the default filter). Make the two sides comparable: a few lines in `scripts/test_nextest_profile_partition.py`. | open |
| #1778 | the expected trap message said `domain in expand`; the fixture's rank-increasing route now traps as `insert` after the #1547/#1590 split. The issue records that the diagnostic itself is intact. | closed |
| #1777 | asserts `ExecutionValue::Tensor` on a scalar cotangent. #1741 made scalar cotangents scalars by spec (`spec/06` section 2.1) and `issue_1741_scalar_gradient_api` pins the new rule. The issue states the gradient value itself is correct. | open |
| #1808 | two slice-B2h receipts matched the host body by its full signature, so #1799's added `chelis_rng_state` parameter read as a changed lane. The issue's own correction records that the ordering property was intact throughout. Repaired by #1810: 43 changed lines in one test file, plus registering the targets in the per-PR selection. | closed by #1810 |

### 9.3 A real defect. Compiler behaviour changed for the worse, or a policy was broken. (3 confirmed, plus half of one)

| item | what is actually wrong | evidence |
|---|---|---|
| #1775 | `grad` of a runtime-symbolic window lost the extent source for one output axis; the backward DAG collapses to rank zero and fails verification. | repaired by #1780: 71 changed lines in `crates/chelis-ir/src/lower.rs` plus two new regression files, 434 insertions total. Not a test edit. |
| #1779 | `chelis build --target c` of a `[nn, 1]` column reshape under `vmap` is refused at lowering: `every staged graph input needs exactly one declared or host producer`. A program that built no longer builds. | diagnostic has exactly one definition site, `crates/chelis-ir/src/host/staged.rs:154`, in a file created whole by #1693 |
| #1746 | a result-only bounded dtype constraint no longer actualises the tensor cast target; lowering rejects with `dtype 'a non-primitive cast target expression' on a 'cast' target`. This is the exact acceptance case #1564 was closed on. | measured: passes at #1724's own merge commit and at `062c29c19`, fails at `22b193cf7` |
| #1787, second half | `examples/dropout_fixed_stream.ch`, shipped into the executable corpus by #1733, cannot emit C because seeded dropout kernels are unimplemented (#1192, open). The census at `issue_1739_diagonal_runtime_bound.rs:548` calls `emit_c` on every `.ch` in `examples/`, so it panics rather than merely miscounting. | source read; #1192 confirmed open |

#1787's *first* half belongs in 9.1: `EXECUTABLE_PHASE_0_EXAMPLES` is pinned to 31 and `examples/`
now holds **36**. Five examples have been added since that constant was written, by #1307
(`count_bool_device_entry`), #1733 (`dropout_fixed_stream`, `dropout_staged_claim`), #1793
(`resource_target_cpu`) and #1809 (`signed_seed`). Four separate pull requests added an example
while the census was red and could not see it.

### 9.4 Undetermined, and this is where the weight sits (7 rows in one issue)

#1776 carries nine rows; two are #1746's and one is #1779's. The remaining six have a measured
first bad commit (#1693) and **no root cause**:

| row | failure text | reading |
|---|---|---|
| `cli::build_hip_emits_pad_kernel` | `HIP pad build must emit the pad kernel source` | likely real |
| `cli::build_hip_emits_shrink_kernel` | `HIP shrink build must emit the shrink kernel source` | likely real |
| `cli::build_c_with_seed_uniform_like_succeeds` | `expected generated C to route the source seed 7 through the effective seed wrapper` | unknown |
| `redteam_731_adversarial::redteam_baked_seed_deterministic_and_cross_lane` | `expected the baked effective-seed wrapper argument 7ULL` | unknown |
| `issue_1222_root_alias_ownership::conditional_over_existing_bindings_claims_no_third_allocation` | `got 3 retain(s) and 5 release(s)` | likely real |
| `issue_1222_root_alias_ownership::seeded_block_returning_an_outer_binding_claims_no_second_allocation` | `got 3 retain(s) and 4 release(s)` | likely real |

Two hypotheses, both unproven and stated as hypotheses:

- **The HIP rows.** #1693 added `RiscOp::Mod => return Err(Self::remainder_unsupported(node))` to
  `crates/chelis-backend-hip/src/emit.rs`, a new device-lane rejection carrying
  `unimplemented_rejection!(1277, "checked integer remainder has no device kernel")`, and routed
  two new ops (`CheckedReshapeExtent`, `CheckedUnitAxis`) through the same match. If pad or shrink
  index arithmetic now lowers through `Mod`, the HIP lane has *lost the ability to compile
  programs it used to compile*. That would be a capability regression, not a stale string, and it
  is adjacent to #1786's HIP-lane drift.
- **The `issue_1222` rows.** The counts move from an expected 2 retains / 4 releases to 3 and 5,
  and 3 and 4. Both remain balanced, so this is an extra allocation and clone pair rather than a
  leak. The test's whole subject is that a conditional over existing bindings claims **no third
  allocation**, so an extra allocation is the defect the test exists to catch, not an incidental
  expectation.

The seed rows are genuinely unknown. The emitter still contains
`CHELIS_EFFECTIVE_UNIFORM_SEED({seed}ULL)` at `crates/chelis-backend-c/src/emit.rs:3487-3491`, so
the string did not move; either the program now selects a different lane, or the seed stopped
reaching the random op. The second reading would be the #703 class, and the test carries a negative
assertion for exactly that (`!c_src.contains("chelis_uniform_sample_f32(0ULL")`).

### 9.5 A defect in the CI change itself, not a break it let through (2)

| item | what is wrong | owner |
|---|---|---|
| #1781 | the census assertion #1736 added compares an `--ignore-default-filter` listing against a default-filtered one, so it cannot pass. It landed into a class #1723 had made nightly-only 36 minutes earlier. | #1736 |
| #1819 | #1723 created `full-workspace` with `timeout-minutes: 60` for `cargo nextest run --workspace --profile ci-full --ignore-default-filter`, 10,365 tests across 738 binaries. The job it replaced ran the narrower `--profile nightly` in a 30-minute budget. The new one has never finished: cancelled at 65 minutes on the nightly `34558777344` and on dispatch `34603833002`. | #1723 |

### 9.6 The count, and why "cheap" is the wrong reassurance

| class | items |
|---|---|
| regenerate, re-pin, or re-freeze | **5** |
| re-target a brittle assertion | **4** |
| real defect | **3 confirmed** (+ the second half of #1787) |
| a defect in the CI change itself | **2** |
| the nightly umbrella (auto-opened) | **1** (#1783) |
| undetermined, all inside #1776 | **6 rows** |

So of the **15 issues**, roughly **9 are cheap repairs** (five frozen-artifact updates and four
re-targeted assertions), **3 to 4 are real defects**, **2 are defects in the CI change**, and one
issue still carries six rows with a measured first bad commit and no mechanism.

That split is reassuring about repair effort and about nothing else. Every one of the cheap items
blocked something expensive while it sat:

- #1818's one-line digest took the whole **`Dtype Phase 0-3 Oracle`** red on `main`;
- #1817's stale freeze takes **`runtime_representation_oracle.py --phase 1`** red on every head
  carrying #1733, which is every head since 10:01 UTC;
- #1808's re-targeted receipt took **`runtime_extent_oracle.py --phase b`** red, so no phase-b
  shortfall reading was possible and every in-flight #1277 slice inherited the failure;
- #1742's tuple drift took **both phases** of the same oracle red, which meant the receipts that
  #1375's closing comment and #1510's evidence cite were not reproducible on `main`;
- #1781 sits in the job that is supposed to prove the nextest partition is complete.

Four of the five acceptance oracles this repository relies on have been red on `main` at some point
in the last day, and in every case the repair is a handful of characters. A one-line repair that
blocks an acceptance oracle for a day is a cheap *fix*, not a cheap *failure*. The expensive part
was never the edit; it was that nothing reported the edit was owed.

---

## 10. The ledger: how many issues this cost in total

**15 issues** trace to the CI diet as of 2026-09-11 14:35 UTC. Section 5 is the earlier snapshot at
12; #1817, #1818 and #1819 were filed at 14:26 to 14:30 and are the authoritative additions. The
count is still live: #1818 and #1819 were the two gaps this analysis named as unfiled, and they
were filed within the hour.

| # | issue | state | class | caused by |
|---|---|---|---|---|
| 1 | #1746 | open | real defect | #1693 |
| 2 | #1747 | closed | re-pin | #1693 |
| 3 | #1775 | closed | real defect | #1693 |
| 4 | #1776 | open | mixed; 6 rows undetermined | #1693 |
| 5 | #1777 | open | re-target | #1750 |
| 6 | #1778 | closed | re-target | #1547/#1590, surfaced |
| 7 | #1779 | open | real defect | #1693 |
| 8 | #1781 | open | defect in the CI change | #1736 |
| 9 | #1783 | open | nightly umbrella (auto) | n/a |
| 10 | #1787 | open | half re-pin, half real defect | #1770 x #1307; #1733 |
| 11 | #1789 | open | re-pin | #1773 |
| 12 | #1808 | closed | re-target | #1799 |
| 13 | #1817 | open | re-freeze | #1733 |
| 14 | #1818 | open | regenerate a digest | #1749, #1751, #1759, #1758 |
| 15 | #1819 | open | defect in the CI change | #1723 |

**Five closed, ten open.** Twelve are breakages that escaped the gate; two are defects in the CI
change itself (#1781, #1819); one is the auto-opened nightly umbrella (#1783).

Against the 48 issues filed in the window, that is **31%**. The other ~33 are red-team semantic
findings from the review rounds on the in-flight #1277 slices and would have been filed regardless.

### 10.1 Cost beyond the issue count

- **28 pull requests merged** in 18.8 hours; at least **10 of them (36%)** left a test red that
  their own CI could not run, and **27 of 28 (96%)** merged onto a tree that already had invisible
  failures.
- **Five of the 28 merges are already repairs** of issues from this same window (#1769, #1780,
  #1807, #1810, #1809). That is 18% of throughput, before the eleven #1693 failures, #1817, #1818
  and #1781 have been touched.
- **Four of the repository's acceptance oracles have been red on `main`** at some point in the last
  day: Dtype Phase 0-3 (#1818), runtime representation Phase 1 (#1817), runtime extents phase a and
  b (#1742, #1808), and the full workspace, which cannot report at all (#1819).

---
## 11. Which job actually catches these, and what kind of thing each job catches

Sixteen issues, sorted by the mechanism that detected them rather than the person who filed them.

### 11.1 By detecting job

| job | issues detected | what it runs |
|---|---|---|
| **Typecheck Level Generalization Oracle (4 shards)** | **7**: #1746, #1747, #1775, #1776, #1777, #1778, #1779 | `cargo nextest run --workspace --profile ci-full --ignore-default-filter --features chelis-types/generalize-sweep-oracle --partition hash:N/4` |
| Dtype Phase 0-3 Oracle | 1: #1818 | `dtype_phase3_oracle.py`, whose preflight checks frozen test-definition digests |
| Runtime Representation Phase 1 Oracle | 1: #1817 | `runtime_representation_oracle.py --phase 1`, whose last leg compares a frozen test selection |
| Surface nightly status | 1: #1783 | opens one umbrella issue saying the nightly failed |
| **Full Workspace (Linux)** | **0** | the job designed for exactly this. Cancelled at 60 minutes having run 1,714 of 10,365 tests. It is itself #1819 |
| Script Integration Tests, Backend Sanitizers, Faithful Observation Phase 2, Compiled Value Ownership Phase 2, Integration Support | 0 | all passed |
| **no CI job at all** | **5**: #1781, #1787, #1789, #1808, #1820 | found by people running full local suites while validating unrelated branches |

Two observations follow, and the second is the one that matters.

**The generalization oracle is accidentally the workspace suite.** It is a feature-flag sweep, not a
coverage job. It happens to list the whole corpus with `--ignore-default-filter`, and it happens to
be partitioned four ways, so it finishes in 15 to 43 minutes per shard. Its unsharded twin,
`full-workspace`, runs a nearly identical selection and catches nothing because it never completes.
Seven of the eleven CI-detected issues came from a job whose stated purpose is something else.

**No CI job reported them, but only one was actually hidden by a job that should have caught it.**
Four of the five are simply younger than the last nightly, which started 2026-09-11 03:31:44 UTC on
`e813415d0`: #1787's cause merged 04:51, #1789's 07:06, and #1808's and #1820's shared cause 12:13.
No nightly has run on a commit that carries them. They were found by people running suites on
demand, which beat a daily cadence; that is an argument for #1824 and for making a branch dispatch
usable, not evidence about `full-workspace`.

**#1781 is the one that `full-workspace` genuinely hid.** Its test was added by #1736 at 18:04 UTC
on 09-10, so it was present in the 03:31 nightly. `ProfilePartitionTests` runs as a *step* in
`full-workspace`, after the nextest run, and the job was cancelled at 60 minutes before reaching it.
That is the direct, measurable cost of #1819.

**What `full-workspace` uniquely covers is small, and it is exactly what has never run.** The
generalization lane's selection is a strict subset of `full-workspace`'s, differing by four tests:
`chelis-cli::stdlib_typecheck_cache_concurrency` (3) and
`issue_1293_redteam_round4::recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c` (1). Beyond
those, `full-workspace` owns two steps no other job runs: the `ProfilePartitionTests` census and the
`chelis_std_self_test_corpus` `--ignored` run. In the 2026-09-11 nightly the corpus step exited 143
when the runner was torn down, and the 2026-09-10 nightly also failed, so it has not completed since
#1723 restructured the workflow on 09-10. The unique coverage and the never-executed coverage are
the same set.

### 11.2 By what the repair is

Classes as defined in section 9, with #1820 added and #1781 counted once in the CI-change class.

| class | count | issues |
|---|---|---|
| A. frozen artifact: regenerate, re-pin, re-freeze | 5 | #1747, #1789, #1817, #1818, #1787 (count half) |
| B. brittle assertion, re-target it | 4 | #1777, #1778, #1808, #1820 |
| C. real defect | 3.5 | #1746, #1775, #1779, #1787 (emit-C half) |
| D. defect in the CI change itself | 2 | #1781, #1819 |
| E. undetermined | 1 issue, 6 rows | #1776 |

### 11.3 The cross-tab, which is the actual answer

| detecting job | A frozen | B re-target | C real | D CI defect | E undetermined |
|---|---|---|---|---|---|
| Generalization Oracle | 1 (#1747) | 2 (#1777, #1778) | **3** (#1746, #1775, #1779) | 0 | 1 (#1776) |
| Dtype Phase 0-3 | 1 (#1818) | 0 | 0 | 0 | 0 |
| Runtime Representation Phase 1 | 1 (#1817) | 0 | 0 | 0 | 0 |
| Full Workspace | 0 | 0 | 0 | 1 (itself, #1819) | 0 |
| no job; found by a human | 2 (#1789, #1787a) | 2 (#1808, #1820) | 0.5 (#1787b) | 1 (#1781) | 0 |

Three things this makes visible:

1. **One job found every real defect.** All three confirmed C-class issues came from the
   generalization oracle. No other job has produced one. If that job were also removed, nothing in
   continuous integration would be finding real compiler regressions at all.
2. **The frozen-artifact oracles are tripwires on their own inputs, and nothing else.** Dtype Phase
   0-3 catches Dtype Phase 0-3's digest; Runtime Representation Phase 1 catches its own frozen
   selection. Each is a complete detector for one artifact and a detector for nothing else, which is
   why two more frozen-artifact drifts (#1789, #1787's count) were found by humans instead: no
   oracle owns those two artifacts.
3. **The cheap classes dominate the count and the expensive class dominates the risk.** A and B
   together are 9 of 16, and every one of them is characters-to-a-line of repair. C is 3.5 issues
   and includes a lowering regression that needed 71 changed lines in `chelis-ir/src/lower.rs` to
   fix (#1775 by #1780). Counting issues makes the situation look like paperwork; counting the
   repair distribution shows that the one job doing real detection is the one nobody designed for
   the purpose.

### 11.4 The nightly's reporting resolution is one bit

Even for the seven the generalization oracle detected, no automation named them. The only artifact
the nightly produces on failure is #1783:

> The **Linux Extended Validation** nightly failed. Run: <url>. Commit: `<sha>`.
> Auto-opened on failure; auto-closes when the nightly next passes.

No job names, no test names, no counts. Every one of the sixteen issues was written by a person who
opened the logs or ran the suite themselves. The nightly detects; it does not report. That is a
separate gap from #1819 and it is not tracked.

---
## 12. Addendum: the #1823 dispatch, and what it actually found

Written after sections 1 to 11. It corrects one of their conclusions.

### 12.1 The partition dispatch

`workflow_dispatch` [34615787779](https://github.com/Chelis-Lang/chelis/actions/runs/34615787779),
2026-09-11 15:23 UTC, head `a466d5f6c` (PR #1823, the four-way partition of `full-workspace`).

| shard | outcome | detail |
|---|---|---|
| **3/4** | **finished, 25 min** | `Summary [996.650s] 2631 tests run: 2625 passed, 6 failed` |
| 1/4 | cancelled at 65 min | blocked on `issue_1314_json_bigint` x2, each **> 2700 s** |
| 2/4 | cancelled at 60 min | reached test **2697 of 2698**, then blocked on `csv_io_end_to_end_is_exact_and_byte_stable`, **> 2940 s** |
| 4/4 | cancelled at 60 min | reached **2616 of 2617**, then blocked on `json_io_end_to_end_is_exact_and_byte_stable`, **> 2820 s** |

Shard 3/4 drew none of the pathological rows and did a quarter of the corpus in **16.6 minutes** of
execution. Shards 2/4 and 4/4 died on their **last test**.

### 12.2 The conclusion this corrects

Sections 1, 4 and 7 treat the `full-workspace` failure as a budget problem, and section 9.5 files
it as a defect in #1723's job sizing. That is incomplete. **The binding constraint is four tests,
not the budget.**

| row | 2026-09-10 13:54 (run 34485518418) | 2026-09-11 03:31 nightly | factor |
|---|---|---|---|
| `csv_io::csv_io_end_to_end_is_exact_and_byte_stable` | **13.889 s** | 2340.732 s | **168x** |
| `json_io::json_io_end_to_end_is_exact_and_byte_stable` | **8.093 s** | 1676.297 s | **207x** |
| `json_io::json_io_missing_path_fails_eval_loudly` | **6.210 s** | 1080.867 s | **174x** |

Both measurements are hosted `ubuntu-latest` at the same parallelism. It reproduces in the
default-feature dispatch as well as the feature-enabled nightly lane, so it is not the sweep flag.

Two consequences:

- **The four-way partition is unproven but not refuted.** Shard 3/4 is the supporting measurement:
  a quarter of the corpus fits the existing 60-minute budget with room. Nothing in the dispatch
  argues against the design.
- **No budget would have fixed #1819.** Four rows at 45-plus minutes each, running two at a time,
  is most of an hour before the rest of the corpus is considered. The 5-hour option section 8
  costed would not have rescued it either.

`.config/nextest.toml` sets **no `slow-timeout`** in any profile, so nextest warns indefinitely and
never terminates a row. That is the amplifier: one pathological test converts into a total loss of
signal rather than one red line.

### 12.3 Where the cost is

Contradictory at first, now resolved. chelis#1829's `sample` shows the host interpreter recursing
through `EvalContext::eval_app` / `eval_list` / `eval_expr` with no lowering frames. A separate
observation of the test spawning `chelis check` suggested the front end. Watching the live process
tree at the bad endpoint settles it: **`chelis check` completes in about one second; `chelis eval
--file` runs for minutes and is the process consuming the time.** The front-end sighting was a fast
preliminary phase caught at one second old.

This is preliminary: a process snapshot says what was running when it was read, not where the time
went. A timed standalone measurement separating the two phases is queued.

### 12.4 Issue ledger changes

Section 10's ledger of 15 becomes **17**:

| issue | what | note |
|---|---|---|
| **#1829** | the 170x to 207x regression | filed 16:32 UTC by the `phase-b-pass-closure` session; parented to **#828** (eval interpreter performance), which the evidence in 12.3 confirms is the right parent; placed as a **Tier 2 release-blocking row on #1362**, with #1784 carried under it as the likely same defect |
| **#1831** | no `slow-timeout terminate-after` | the amplifier in 12.2; genuinely CI-only |

**#1830 was a duplicate of #1829**, filed by this analysis without checking for an existing issue
first, and is closed. Its measurements are folded into #1829 as a comment. **#1824** (decide the
per-pull-request selection) is a decision issue rather than a defect and is counted separately.

Section 6 should also be read with a correction: **#1784 is probably not an independent
load-sensitivity flake.** It records the stdlib `tests/io/csv.ch` rows sitting within about three
seconds of a 30-second cap. `csv_io_end_to_end` went 13.9 s to 2340.7 s over the same window, so
#1784 is very likely this defect surfacing under a different harness.

### 12.5 Regression window and culprit

`f5c2a1ea8` (the fast run's head) and `e813415d0` share merge-base **`df46fae13`** (#1717,
2026-09-10 10:53 UTC): **29 commits**. Confirmed independently on this workstation: `df46fae13`
measures `Summary [7.999s]`, 8.6 s wall, for `csv_io_end_to_end_is_exact_and_byte_stable`.

**The bisect converged on `22b193cf7` — PR #1693 again.** Parent `062c29c19` (#1736), the CI change,
so two adjacent measurements isolate #1693 rather than a range:

| commit | `csv_io_end_to_end` | bare 13-line JSON program, `chelis check` | bare program, `chelis eval --file` |
|---|---|---|---|
| `df46fae13` | `Summary [7.999s]` | 2.11 s | 1.80 s |
| `062c29c19` (parent) | `Summary [8.764s]`, PASS | 2.10 s | 1.80 s |
| `22b193cf7` (#1693) | **> 480 s** | 2.19 s | **> 300 s** |

The front end is flat. **Evaluation carries all of it, at least 167x, on a bare user program.**
Both bad-end figures are bounds; neither run was observed to complete.

**So section 3's ranking understates #1693.** It is the first bad commit for eleven of the thirteen
nightly failures *and* for this regression, which is the one that takes the workspace job down
entirely. One pull request accounts for both.

**Correction, and it was mine.** Earlier revisions of this section asserted that none of the 29
commits was a #1277 or runtime-extent change, which excluded #1693 from suspicion. That was wrong.
The check behind it was `git log --oneline ... | grep -icE "1277|runtime extent"`, which greps
squashed **subject lines** only; #1693's subject is "Preserve computed reshape claims and broadcast
unit checks through calls" and names neither. Grepping full commit **bodies** returns 10 matching
commits, and #1693's own body carries 36 commits of which 18 mention extents, adding
`changelog.d/checked_extent_transport.fixed.md` and
`crates/chelis-ir/tests/runtime_extent_checked_transport.rs`.

The exclusion was passed to the bisect agent and to the session doing the #1277 phase-b closeout, as
a fact. The agent did not prune on it and landed on #1693 by measurement, so it cost nothing here.
It would have sent anyone who did prune straight past the answer. An assertion derived from a check
that cannot see the thing it rules out is not evidence, and this section previously presented one as
if it were.

---

## 13. Closing: what the question turned out to be

Written at the end of the day's work. **This section closes the document**; later state belongs on
the issues, not here.

### The question, and the layered answer

The document opens asking which issues were caused by pull requests merging red against tests moved
to nightly. The answer turned out to have three layers, only the first of which was the question
asked.

1. **The CI diet did not cause the defects. It removed the reporting.** #1723 and #1726 took two
   whole-workspace nets off the pull-request path within 36 minutes of each other. Nothing broke
   because of that; things broke and nothing said so.
2. **One pull request caused most of them.** #1693 (`22b193cf7`) is the first bad commit for six
   issues — #1746, #1747, #1775, #1776 (nine rows), #1779 and #1829 — spanning lowering, ownership
   accounting, HIP emission, a schema guard and the evaluator. It merged four hours after the
   coverage that would have caught it was removed.
3. **#1693 carries two independent mechanisms, and the worst one was not its fault.** The
   `CheckedProgram::compose` change routed the interpreter into an unmemoized exponentially
   branching probe (chelis#1829, 170x to 207x). The 676-line `host/staged.rs` region is a separate
   mechanism and is what #1775 and #1779 name. And the exponential itself **pre-exists #1693**: on
   its parent, thirteen mutually recursive defs with fan-out two already hang `chelis eval`. #1693
   removed an accidental guard. That underlying defect is now chelis#1835.

So: a coverage change made a regression invisible; the regression turned out to be a trigger; and
the trigger exposed a standing defect nobody had met. Each layer needed the one before it to be
found.

### What shipped

- **PR #1823**, merged `81bb57e52`. Partitions `full-workspace` four ways and normalises both sides
  of the census assertion onto the `--ignore-default-filter` basis. Red-team round 1: no P0, no P1,
  two P2 and four P3, all closed or accepted.
- **chelis#1781 closed**, verified byte-identical on `main` rather than inferred from the merge.

### What is open, and who has it

| issue | state |
|---|---|
| #1819 | open deliberately: the partition is necessary and not sufficient until #1829 lands |
| #1829 | owned by the diagnosing session, repair in flight, receipt shape agreed |
| #1835 | design posted, implementation gated behind #1829 |
| #1817, #1818, #1777, #1787, #1789, #1820, #1831 | one pull request in flight |
| #1838 | filed from #1789's second drifted row: `insert` reports its size error as `expand size` |
| #1824 | a decision for the maintainer, costed |
| **#1746, #1776, #1779** | **unowned**, and confirmed not fixed by either the #1829 repair or the #1835 design |

### The thing worth keeping

Not the partition. **Five instances in one day of claims outrunning the evidence already in hand**,
across two sessions and four agents, including twice from the orchestrator. None was a skipped
check; every one was a compression step after the check. They split two ways — summary drift, which
re-reading against the raw output catches, and coordinate rot, which it cannot — and the fix that
covers both is to quote the source rather than describe it.

The sharpest single form: **an asserted negative over a set the receiver cannot sample.** "None of
these 29 commits is #1277 work" cost more to verify than the bisect it was meant to inform, so
trusting it was rational and the brief instructed it. It was false, and it surfaced only because the
bisect happened to land on the excluded commit.

---
## 14. Addendum, 2026-09-11 evening: what #1693 actually did, and the fourteen minutes

Everything in sections 1 to 13 stands. This section adds the mechanism behind the
ranking in section 3, which that section could only establish by measurement, and
records what changed on the ledger since.

### 14.1 The fourteen minutes

`062c29c19` is the commit that moved these targets off the per-pull-request path. It
merged at **14:04 EDT** on 2026-09-10. `22b193cf7` (#1693) is its **immediate child**
and merged at **14:18 EDT**, fourteen minutes later.

That is not a coincidence worth a footnote; it is the finding. The single pull request
that accounts for more breakage than every other merge in the window combined is the
very next one to land after the checks that would have caught it stopped running. Nine
integration rows across five test targets went red, in three unrelated surfaces, with
no red check anywhere.

### 14.2 What #1693 is

Title: "Preserve computed reshape claims and broadcast unit checks through calls."
Body: **36 commit messages**, 74 files, +6835/-519, one squash.

The subject describes the first of at least four separable work streams in it. The one
that caused the breakage is a different stream, about ten commits deep: **staged host
sources**, which introduced a new 676-line `crates/chelis-ir/src/host/staged.rs`.

Before #1693 a tensor-returning call went one of two ways. It became a *tensor helper*,
one dataflow graph the backend compiles into a kernel, or it stayed on the host lane as
ordinary scalar code. #1693 added a third: partition one function into alternating host
and tensor segments, so a value computed by host scalar code can reach the tensor graph
with its extent claim still attached. The motivating case is real. A `reshape` whose
target size is read at runtime needs that computed size to arrive in the graph carrying
the proof of what it is.

Every consequence below follows from introducing that third path, and every one of them
is visible in the diff.

### 14.3 Five consequences, all measured

**1. Tensor-helper lowering is now refused whenever the callee stages.**
`lower_app_host_expr` gained `&& !callee_has_stages` at both of its lowering sites. The
tensor helper is exactly what the HIP emitter consumes, so any operation that moved onto
the staged path stops reaching the device backend at all. `pad` and `shrink` now emit
**zero** device kernels and run on the CPU. The HIP emitter still contains `kernel_pad`
(`chelis-backend-hip/src/emit.rs:1474-1481`); nothing routes to it. Filed as
**chelis#1842**.

**2. A new hard rejection with no fallback.** `host/staged.rs:154` requires every `Load`
in a staged graph to be exactly one of {declared external input, host-produced}, an XOR
written `if parameter == source { Err(...) }`. When both or neither hold, lowering fails
outright with "every staged graph input needs exactly one declared or host producer".
`vmap` over a symbolic column trips it. That is chelis#1779 and chelis#1775.

**3. Random draws had to stop being baked**, and the commit says why:

> A source may itself draw from Random. Each executed tensor segment must continue the
> live handled stream, rather than baking the draw count inferred before those source
> expressions have executed.

So the seed moved from an op-site constant into the live invocation-local RNG state,
and `CHELIS_EFFECTIVE_UNIFORM_SEED(7ULL)` became `(0ULL)` with the 7 written to the
state. Note for anyone re-deriving this: `COMPILED_HANDLER_OWNED_SEED = 0` in
`lower.rs:17` explains the *head* spelling but is not #1693's mechanism, because it
arrived the next day with #1733.

**4. Host-lane routing changed for two node kinds.** `If` left the unconditional
keep-on-host list and gained literal-condition folding; `HandleEffect` was added to it.
That is precisely the two `issue_1222` retain/release rows: `if true` now folds and its
two arm clones genuinely vanish, while a `with seed` scope that used to disappear
entirely now materializes and emits one extra retain with a matching release.

**5. And, from an unrelated commit in the same squash**, `runtime/mod.rs` changed
`program: Some(program)` to `program: Some(kernel_program.as_ref().unwrap_or(program))`,
where `kernel_program` is `CheckedProgram::compose(library, program)`. Before, the
interpreter's program held new code only, so asking whether `parse_value` is a compiled
kernel found no such definition and returned at once. After, every `Std.Io.Json`
definition is visible and the interpreter walks an unmemoized, exponentially branching
kernel-decision probe over roughly fifty mutually recursive parser definitions. That is
chelis#1829, the JSON slowdown. #1693 did not author the exponential; it removed an
accidental guard, and the composition itself is a correctness fix.

**None of the five is a bug in the ordinary sense.** Each is a deliberate change with a
defensible reason stated in its own commit message, and several are fixes. What made
this a multi-issue event is that they landed together, under a subject line naming none
of them, in a squash whose body a reviewer would have to read thirty-six messages deep,
fourteen minutes after the last check that would have seen any of them stopped running.

### 14.4 The nine chelis#1776 rows, now fully classified

All six previously unattributed rows measure to `22b193cf7`. Method was not a bisect: a
four-signal probe reading every row off one `cargo build -p chelis-cli --bin chelis`, at
six points over `12c04c66a..0e858f184`, with a passing row as a negative control that
held its expected value at every point, and tri-valued signals so an uncompilable commit
could not read as the regression.

| row | class | disposition |
|---|---|---|
| `build_c_with_seed_uniform_like_succeeds` | stale pin | apply what #1809 did to its twin |
| `redteam_baked_seed_deterministic_and_cross_lane` | **already fixed** | #1809 repaired this one and left the twin |
| `build_hip_emits_pad_kernel` | **real regression** | chelis#1842 |
| `build_hip_emits_shrink_kernel` | **real regression** | chelis#1842 |
| `conditional_over_existing_bindings_claims_no_third_allocation` | stale pin | literal-`if` fold |
| `seeded_block_returning_an_outer_binding_claims_no_second_allocation` | stale pin | `with seed` materializes |
| `result_constraints_actualize_bounded_tensor_targets` | not re-measured | chelis#1746 |
| `result_only_cast_instances_do_not_share_precision` | not re-measured | chelis#1746 |
| `vmap_over_symbolic_column_builds_without_symbolic_dim_ice` | **real regression** | chelis#1779 |

**Correction to an earlier reading in this document.** The `3 retain(s) and 5 release(s)`
and `3 retain(s) and 4 release(s)` numbers looked like an over-release reaching generated
code, and I treated that as a possible soundness defect. **They are not.** Both emitted
programs were linked and run under the runtime's ownership ledger, whose `release_header`
traps an invalid release. Both print correct values and exit 0, and every allocation nets
to exactly zero. A runtime condition still emits `(5,6)`, identical to the good side. No
`soundness` label is warranted.

`#1809` is worth naming separately: it repaired one of two byte-identical assertions and
left the other standing. A repair that fixes its witness and not its class is this
document's subject in miniature.

### 14.5 Ledger changes

**Final state at 2026-09-11 20:40 UTC.** Nine issues closed, each verified on `main`
rather than inferred from a merge, because merging closes nothing here and a merged pull
request is not proof its issue is resolved.

| closed | resolved by | evidence taken on `main` |
|---|---|---|
| #1781 | #1823 | byte-identical verification |
| #1787 | #1834 | main's own `Fast Tests (Linux)`, row passing at 5.759 s, the job's nextest line naming the target |
| #1817 | #1828 | oracle's own `frozen_manifest` and selection code, 129 frozen tests matched |
| #1789 | #1839 | main's `Fast Tests`, target newly registered so it executes rather than being listed |
| #1777 | #1839 | same, plus `spec/06` §2.1 settling that the scalar carrier is correct |
| #1820 | #1839 | 19/19 by name against `33001549a`, denominator derived from the 21 pre-fix failures minus the two #1776 rows |
| #1831 | #1839 | `scripts.test_ci_cadence` 11/11 on main's tree, lock proven in both directions |
| #1845 | #1849 | `LOUD UNSUPPORTED PHASE 3 ORACLE: PASS`, and red on the pre-repair tree |
| #1818 | #1839 **and** #1849 | `PHASE 3 ORACLE: PASS`, exit 0, both blocking rows executing at 6/45 and 8/45 |

**#1818 is the one worth reading twice.** It needed two repairs, and the second was
invisible until the first landed: #1839 moved the digest the issue reported, the oracle
then stopped at a *different* leg, and that leg became #1845. The reason the closure is
real rather than nominal is that #1839's author **narrowed its own claim** instead of
reporting an oracle exit nobody had observed. A pull request that had claimed `ORACLE:
PASS` on a repaired leg would have closed #1818 three hours early and left it red.

**Merged:** #1823, #1839, #1849 from this stream; #1828, #1834, #1846 from two others.

**Filed and open:** #1842 (the chelis#1693 HIP device-lane regression, the only real
capability loss in the set), #1847 (seven stranded prose citations to a closed issue,
five in contract documents), #1850 (a third narrow-float diagnostic pinned nowhere, with
`F8e4m3` and `String` reaching it live), plus #1841 and #1844 as tracked residuals.

**Still open and unowned:** #1842, #1850, #1847, #1779, #1746, and the remainder of
#1776.

**Two defects landed during this investigation, from #1828, in exactly the shape the
investigation was about.** A stale frozen selection and four stale pinned diagnostics,
both invisible per pull request, both found only by an agent rebasing across them. That
is the strongest available evidence that chelis#1824 is the open structural item: not one
bad merge on 2026-09-10, but a selection nobody has decided the purpose of, which keeps
admitting the same class a day at a time.
### 14.6 One correction to section 12

Section 12 reads the `heavy-e2e` dispatch queue as runner contention. It is not. The
workflow declares `concurrency: group: linux-extended-${{ github.ref }}` with
`cancel-in-progress: false`, so two dispatches on the same branch **serialize**: the
second sits at `status=pending` with zero jobs until the first finishes, however idle the
runner pool is. Observed with 11 jobs active repo-wide. Compare `headBranch` across
recent runs before blaming the pool, and note that dispatching a third run on that branch
queues it behind both.

## Appendix: reproduction commands used

```sh
# Worktrees at #1693's parent and at #1693, each with its own .venv and CARGO_TARGET_DIR
git worktree add --detach ~/chelis-worktrees/ci-triage-base 062c29c19
git worktree add --detach ~/chelis-worktrees/ci-triage-1693 22b193cf7
uv venv --python 3.11

# Default configuration (six rows)
cargo nextest run -p chelis-cli \
  --test cli --test issue_1222_root_alias_ownership --test redteam_731_adversarial \
  --no-fail-fast \
  -E 'test(=build_c_with_seed_uniform_like_succeeds) + test(=build_hip_emits_pad_kernel) +
      test(=build_hip_emits_shrink_kernel) + test(=redteam_baked_seed_deterministic_and_cross_lane) +
      test(=conditional_over_existing_bindings_claims_no_third_allocation) +
      test(=seeded_block_returning_an_outer_binding_claims_no_second_allocation)'
# 062c29c19: 6 passed.   22b193cf7: 6 failed.

# Generalization configuration (issue_1564)
cargo nextest run -p chelis-cli --test issue_1564_bounded_tensor_cast \
  --features chelis-types/generalize-sweep-oracle --no-fail-fast
# 3c0d46137 (#1724): 7 passed.   062c29c19: 7 passed.   22b193cf7: 5 passed, 2 failed.
```

Both worktrees were removed with `git worktree remove --force` and `git worktree prune`.
