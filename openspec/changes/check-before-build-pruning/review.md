# Review Evidence

## Red baseline

The parent revision was `afb419e2db95398f2a43fae10d037158104802b7`.

`chelis check` rejected each invalid direct, transitive effect, and transitive linearity fixture. Cache-disabled, cold, and warm builds accepted each fixture.

This baseline confirmed the chelis#1184 divergence before implementation.

## Mutation review

A fresh audit used a detached worktree and an independent `openai-codex/gpt-5.6-luna` agent.

The audit verified these mutations:

- Moving eval-only removal before semantic checks restored the invalid build acceptance.
- Replacing the transitive worklist with one-hop removal failed the direct unit fixture.
- The failure retained `wrapper`, its `defsig`, `outer`, and its `defsig`.
- Applying the eval-only backend gate before pruning rejected the valid unused eval-only fixture.
- The clean warm build used layered success without a monolithic selected-program check.
- Moving the invalid outside-target file into `src` made the source-selection build fail.

The audit restored every mutation.

## Audit corrections

The first audit found specification, changelog, and branch-invariant gaps.

The implementation and artifacts now include these corrections:

- A direct unit fixture pins arbitrary-depth removal and paired `defsig` removal.
- A linked Reef fixture pins the separate whole-program `tensor_scan` gate.
- The normative text scopes eval-only rejection to the retained compile target.
- The Unreleased changelog contains the breaking migration note.
- Loose C host-surface detection uses the checked post-drop program.
- The source-selection fixture uses a file inside the package tree and outside the target.

A final fresh audit found no implementation or specification inconsistency.

## Focused results

The final audit recorded these results:

- `library_cache_oracle`: 20 passed.
- Transitive-closure unit fixture: 1 passed.
- `process_run_builtin`: 7 passed.
- `issue_257_tensor_scan_host_runtime`: 18 passed.
- Rust format check: passed.
- Strict OpenSpec change validation: passed.

The authoritative focused command is:

```text
CARGO_TARGET_DIR=target/agents/check-before-build-pruning cargo nextest run -p chelis-cli --test library_cache_oracle --no-tests fail --no-fail-fast
```

## Full local gate

Two maximum-parallel gate runs passed 2,083 of 2,085 `chelis-cli` tests. Only two timing-sensitive eval-timeout tests failed under process-start load.

The complete six-test timeout suite then passed in isolation.

The unchanged full gate passed with `NEXTEST_TEST_THREADS=8`. This limit reduced process-start contention and did not skip tests.

The final command used the isolated target and runtime directories:

```text
CARGO_TARGET_DIR=target/agents/check-before-build-pruning \
CHELIS_RUNTIME_DIR=target/agents/check-before-build-pruning/debug \
NEXTEST_TEST_THREADS=8 \
devenv shell -- python3 scripts/gate.py --local
```

## Hosted evidence

Manual CI run `32873124904` tested implementation commit `60ed69671339994f6d88bb7f29151ac145b01f7d`.

The required `Integration Tests (Linux)` check passed. Its dependent Linux workspace tests also passed.

The first overall run found one unrelated policy-test false positive. A workflow comment contained lowercase `devenv shell`, which the literal command scanner rejected.

The comment now uses the product name `Devenv`. The complete 955-test Python script suite passed after this correction.

Run URL:

```text
https://github.com/Chelis-Lang/chelis/actions/runs/32873124904
```
