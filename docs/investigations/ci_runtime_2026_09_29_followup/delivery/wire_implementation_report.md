# Wire probe reuse implementation report

**Status:** Complete, committed locally; no push, PR, CI dispatch, or delegated agent. The primary checkout was read only.

- Worktree: `/Users/robertronan/chelis-worktrees/wire-probe-reuse-20260929`
- Branch: `perf/wire-probe-reuse-20260929`
- Freshly fetched base: `origin/main` at `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`
- Commit: `4bfcf238c3fed0e0c9577d911d39b0d8ee438ab7` (`perf(ci): reuse verified wire publication probe in controls`)
- Working tree: clean. Own `uv venv --python 3.11`; no remaining repository build/test processes according to `scripts/reap_orphans.py` and process inspection.

## Change and proof boundary

I wrote failing C6-derived routing and rejection tests before implementation. The schema stage now carries the SHA-256 of the probe it actually executed. Only the supervised mutation Python selection receives the schema target and digest. The runner accepts a canonical target within this worktree's `target/` containing an existing, nonsymlink probe with exactly those bytes; it validates before execution and again afterward. The supervisor validates at both ends and emits target/digest in its framework packet. The parent verifies that packet and records target/digest in `TestExecution`, which appears in the wire execution evidence. A missing, foreign, redirected, or altered probe rejects.

`LocalPublicationControls.setUpClass` uses the selected target solely for `build_probe`. Its `self.target` remains `target/agents/wire-codec-rustdoc` for fixture Rust expansions. With no override, the standalone control builds in that original target. The verifier still performs the source-bound schema, acceptance, cache/publication, and invocation stages; the codec cold rebuild, baseline, selected/executed checks, and separate invocation target remain intact. No registry or baseline was edited. Four exact CI path rules were added because changed paths initially failed strict classification.

## Local checks

All commands ran in the dedicated worktree with its `.venv`; Cargo used `CARGO_TARGET_DIR=$PWD/target/agents/wire-outer` and `CARGO_BUILD_JOBS=2`. Wall times are `time -p` real seconds where recorded.

| Check | Result |
|---|---|
| `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_wire_runner.SelectedWireProbe test_capacity_census_wire_local.LocalProbeTargetRouting` | 5/5 pass; rerun after final test-only formatting edit |
| `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_wire_runner` | 12/12 pass, 0.64 s |
| `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_wire_local` | 14/14 pass, 104.86 s, including standalone build behavior |
| `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_wire_verifier test_capacity_census_wire_schema.SchemaCases` | 24/24 pass, 0.14 s |
| `cargo nextest run -p chelis-compiler-api --test capacity_census_wire -E 'test(/^wire_schema_numeric_fields_match_the_reviewed_baseline$/)' --no-fail-fast` | **Full local source-bound wire case 1/1 pass**; test 1058.465 s, outer real 1197.89 s |
| Same `cargo nextest` target with `-E 'not test(/^wire_schema_numeric_fields_match_the_reviewed_baseline$/)'` | Remaining 17/17 pass, 0.75 s |
| `PYTHONPATH=scripts .venv/bin/python -m unittest test_ci_change_owned` | 153/153 pass, 10.50 s |
| `.venv/bin/python scripts/ci_change_owned.py classify-paths --from-git --base origin/main`; `git diff --check HEAD^ HEAD` | Pass; all 7 paths classified, no whitespace errors |

The full-case execution receipt is `target/capacity-census-wire-execution.json` in this worktree. It records the selected/executed mutation controls as 135/135, the selected probe target as `target/agents/729-capacity-rustdoc`, and the schema and acceptance probe SHA-256 as `736a1eb38e1d7520135d797ee719ab347c1d4da5a7a4fec9b1fc51eaffbf21e5`. That full-case run preceded the final formatting edit in the test file; production routing code was unchanged, and the new tests passed again afterward. The outer run included a cold build and overlapped an unrelated workspace build, so it does not establish a wall-time saving.

## Exact CI route and limit

The changed paths have these `.config/ci-test-targets.toml` owner routes:

| Paths | Owner job | Cadence |
|---|---|---|
| `.config/ci-test-targets.toml`, `scripts/capacity_census_wire_runner.py`, `scripts/test_capacity_census_wire_runner.py` | `ci.yml` `script-unit` | PR and push |
| `scripts/capacity_census_wire_schema.py`, `scripts/capacity_census_wire_acceptance.py`, `scripts/capacity_census_wire_verifier.py` | `heavy-e2e.yml` `dtype-phase3-oracle` | daily and manual dispatch |
| `scripts/test_capacity_census_wire_local.py` | `heavy-e2e.yml` `script-nightly` | daily and manual dispatch |

`ci.yml` runs `scripts/ci_script_tests.py pr`. Its generated PR roster includes all five new routing tests. The 12 compiler-backed `LocalPublicationControls` methods belong to the census roster, while the new `LocalProbeTargetRouting` class belongs to PR script-unit. A local push-mode planner simulation from the exact base to commit classified all seven paths as `path_rule_owner`, with `selected_packages=[]`, `change_owned=[]`, and `package_expansion=[]`. Thus these Python changes select **script-unit controls on PR CI, not the full wire Rust case**. The full case was executed locally as above; there is no hosted full-case validation for this commit. The orchestrator still owns the fast gate, push, PR, and review.
