# PR #2676 red-team round 1

Reviewed head: `07c286e0b98202858b1f04790cad7fb482d90f52`.
Verdict: **not satisfied; two in-scope P1 findings remain open.**

## P1 — R1: GitHub run provenance uses the wrong commit identity

Class: branch-head versus synthetic-candidate identity confusion. Scope: introduced; prevents the PR's primary reuse behavior.

`scripts/ci_expansion_fast_reuse.py:242-266` queries workflow runs by the synthetic candidate SHA, then requires the run, Fast job, and artifact API `head_sha` fields to equal that SHA. GitHub's PR workflow metadata uses the PR branch head for all three fields. The synthetic candidate is the commit checked out by the job and recorded inside its coverage artifact.

Live reproduction against successful CI run **36331094493**:

- Run, Fast job, and artifact head: `893646161c69d394c4fd504afab7b9314273eb22`.
- Coverage artifact candidate: `42eb8a093ab1be72558691c9fb8b2b103271724b`.
- Fast job conclusion: success. Coverage success: true, 97 executed standing targets.
- Calling `_source_evidence` with that actual coverage candidate returns `None`.

Thus an ordinary successful PR run always becomes fallback, so this change does not remove the repeated executions it claims to address. Changing just the search query is insufficient: job and artifact comparisons have the same defect. Discover source runs with the separately validated branch head and retain exact synthetic-candidate validation on execution evidence.

Reproduction command (from review worktree; network read only):

```sh
PYTHONPATH=scripts .venv/bin/python - <<'PYCODE'
import json
import ci_expansion_fast_reuse as r
repo='Chelis-Lang/chelis'; run_id=36331094493
run=r._gh_json(f'repos/{repo}/actions/runs/{run_id}')
job=next(j for j in r._gh_json(f'repos/{repo}/actions/runs/{run_id}/jobs?per_page=100')['jobs'] if j['name']==r.FAST_JOB)
a=next(a for a in r._gh_json(f'repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100')['artifacts'] if a['name']==r.FAST_RECEIPTS)
c=json.loads(r._archive_member(r._gh_archive(repo,a['id']),'ci-fast/coverage.json'))
print(run['head_sha'],job['head_sha'],a['workflow_run']['head_sha'],c['candidate_sha'],job['conclusion'],c['success'])
print(r._source_evidence(repo,c['candidate_sha'],api=r._gh_json,download=r._gh_archive))
PYCODE
```

Also independently inspected this PR's run 36332872274: Actions head `07c286e0...`, PR merge candidate `7a9ad6e1...`. That run did not execute Fast successfully, so it is corroborating identity evidence, not the positive source proof.

## P1 — R2: Required features do not prove equivalent Cargo configurations

Class: incomplete Cargo feature-equivalence proof. Scope: introduced; the authorization predicate permits omitted expansion-only tests once R1 is repaired.

`scripts/ci_expansion_fast_reuse.py:166-188` compares target `required-features` against the plan and permits groups with empty lists. Fast's unique integration names execute in `cargo nextest ... --workspace --lib --bins --test ...`; expansion executes package-scoped groups. Dependency feature unification can enable features in the workspace invocation that are absent from the package invocation, even when the target's `required-features` is empty. `cargo metadata --no-deps` target admission lists cannot establish the resolved build feature sets.

Executed a normal two-package Cargo workspace fixture with resolver 2: package b depends on a with feature `extra`; a's smoke integration contains `common` and a `#[cfg(not(feature="extra"))]` test `default_only`. Actual nextest listing results:

```text
Fast ['common']
Expansion ['common', 'default_only']
Authorization {'fast_tests': ['a::smoke::common'], 'reused_targets': ['a::smoke'], 'reused_tests': ['a::smoke::common']}
UNSAFE OMISSION REPRODUCED: expansion-only default_only is omitted
```

The authorizer used the fixture's actual Cargo metadata, a complete matching Fast receipt/JUnit, and no target exclusions. It authorized the whole target despite the missing expansion case. This is a routine dependency/feature arrangement, not malformed evidence. Matching test names alone would also be insufficient where features change the implementation under test. Require equivalence of effective Cargo build configurations, or conservatively refuse reuse when that cannot be established.

Exact self-contained reproduction retained outside the repository:

```sh
.venv/bin/python /Users/robertronan/chelis-worktrees/ci-expansion-redteam-probe.py
```

The probe creates and removes a temporary workspace; no reviewed source is modified.

## Executed validation and scope coverage

- `PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_ci_expansion_fast_reuse`: 10 passed. Positive authorization and stale/tampered/partial/failed/mode, missing JUnit, feature admission, ignored/excluded target, old-interface, and digest controls executed.
- `PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_ci_change_owned.ShardingAndExecutionTests scripts.test_ci_change_owned.ReportTests`: 60 passed. Includes whole-target skip without local build/run, summary accounting, authorization mismatch rejection, and the existing disjoint-selection/report controls.
- Live successful-run artifact download and authorizer invocation established R1.
- Actual Cargo/nextest fixture plus direct authorization established R2. No compiler workspace build or full fast-gate rerun.
- Read all seven changed files, the complete PR body/comments, Fast workflow producer, existing receipt producer/executor/report, and Nextest profiles. Legacy candidate routing and trusted summary fail conditions are present. Sanitizer workflow selection is not changed.
- Claim 1 fails on R1 and R2. Claim 2 has tested fallback controls, but cannot establish feature mismatch safety because of R2. Claims 3 and 4 have focused tests and workflow source coverage. Claim 5's feature-equivalence portion fails on R2; disjoint ownership and separate sanitizer routing remain supported.
- No additional out-of-scope defects are reported. Author separately reported hosted workflow-contract failures; I did not diagnose or claim those as independent findings.

Unvalidated: production default-branch dispatch of this workflow, full hosted CI acceptance, and real expansion duration savings. There is no successful dispatch evidence for this PR's new workflow. No promise of exhaustive malformed-input coverage is made.

Final state at 2026-09-27T16:26:22Z: worktree_status verdict FREE; exact reviewed HEAD unchanged; clean (0 modified, staged, untracked, unmerged); no scoped processes. All temporary Cargo fixtures were removed. The report and reproducible probe remain outside the worktree for repair verification. I remain available as the standing reviewer for this round.
