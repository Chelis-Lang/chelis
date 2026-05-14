# Integration Test Suite Trim

Post-release deep cleanup of the integration test suite, the workstream deferred
by `ci_integration_test_runtime_diagnosis.md` ("Deep cleanup (test
consolidation, pareto-optimal-subset trimming, reorganization) is deferred to a
post-release workstream"). This note records the profiling, the categorization
verdict per heavyweight file, and the trim that was applied.

## Profiling method

`cargo build --workspace --all-targets` first (so timing measures execution,
not compilation), then `cargo nextest run --workspace` with per-test timing
extracted from the PASS lines. Workstation has 12+ cores.

Baseline: **3142 tests, ~31s wall** under nextest on this box. The CI's
historical 691s figure is from the old serial `cargo test --workspace --tests`
runner; CI swapped to `cargo nextest` in PR #117. The remaining bottleneck
under nextest is a long tail of heavyweight `chelis-cli` integration tests.

## Root-cause finding

Every test in the heavyweight `ws*` / `rt*` suites that runs **over 1.5s** is a
`chelis check` or `chelis build` invocation against a **production stdlib file**
under `packages/chelis-std/src/`. Every *synthetic-replica* test (a single-file
`.ch` fixture written to a tempdir) runs **under 0.15s**, including the
multi-dtype loops.

A single `chelis check packages/chelis-std/src/nn/linear.ch` invocation takes
~4.4s and produces ~47800 typed nodes: it re-typechecks the entire chelis-std
transitive import graph from scratch every process. There is no cross-process
typecheck cache for a raw `chelis check` on a stdlib file. So the cost is
inherent per-invocation, and the cruft is the **duplication**: the same handful
of production stdlib files were independently checked/built across `wsc_v3`,
`wsc_stdlib_generalization`, `rt3_adversarial`, and `wsa8`.

The synthetic-replica type-system tests are NOT the problem. They are fast,
unique coverage and were kept.

## Production-stdlib invocation duplication map (pre-trim)

| Stdlib file | Checked/built in |
|---|---|
| `nn/linear.ch` | wsc_v3 (check), rt3 (build), wsa8 (build x2 + check) |
| `nn/attention.ch` | wsc_v3 (check), rt3 (check), wsa8 (build + check) |
| `nn/embedding,rmsnorm,silu,gelu,generate` | wsc_v3 only |
| `init/random,kaiming,xavierext` | wsc_v3 only |
| `loss/bce,crossentropy,kldiv,metrics` | wsc_v3 only |
| `optim.ch`, `test.ch` | wsc_v3 only |
| `tensor/reduce.ch`, `nn/conv.ch`, `init/xavier.ch` | wsc_stdlib_generalization only |

## Categorization verdict per heavyweight file

| File | Tests | Verdict | Reasoning |
|---|---|---|---|
| `wsc_v3_stdlib_finish.rs` | 37 | **consolidate** | Sections 1-4 (21 synthetic-replica type-system tests) are fast + unique, kept in place. Section 5 (16 `production_stdlib_*_typechecks`) is the slow mass; moved to a single deduplicated `production_stdlib_typechecks.rs` and deleted here. |
| `wsc_stdlib_generalization.rs` | 20 | **consolidate** | WS-C v2; v3 header says it covers what v2 escalated. Synthetic stub-sig tests kept (fast, unique reduce/conv/xavier shapes). The 3 `production_stdlib_*_sig_file_type_checks` moved to the consolidated file. |
| `rt3_adversarial.rs` | 16 | **consolidate** | Shipped-fix locks for WS-A8 (BLOCKERs inverted to post-fix asserts). Synthetic adversarial tests kept. `production_stdlib_linear_now_builds_clean` (build) is redundant with `wsa8::build_stdlib_linear_succeeds`; `production_stdlib_attention_typechecks_clean` (check) is redundant with the consolidated file. Both deleted. |
| `wsa8_monomorphization_build.rs` | 8 | **trim** | `build_stdlib_linear_succeeds` + `build_stdlib_attention_succeeds` are the unique build-path coverage for monomorphization; kept. `check_stdlib_linear_still_clean_post_wsa8` + `check_stdlib_attention_still_clean_post_wsa8` re-check the same files the consolidated check file covers; deleted. Synthetic build-reject tests kept. |
| `rt4_adversarial.rs` | 25 | **keep** | "Final adversarial sweep" pinning F1-F7 shipped fixes (explicit PR list in the header). All synthetic, all under 0.5s. Genuine shipped-fix regression locks. |
| `wsa5_precision_polymorphism.rs` | 5 | **keep** | All synthetic, under 0.025s. WS-A5 type-system locks. |
| `wsa6_def_annotation_desugar.rs` | 6 | **keep** | All synthetic, under 0.015s. WS-A6 desugar locks. |
| `wsa7_bareref_return_inference.rs` | 6 | **keep** | All synthetic, under 0.013s. WS-A7 inference locks. |
| `rt3a_adversarial.rs` | 20 | **keep** | All synthetic, under 0.05s. WS-A5 RT-3a fixup locks. |
| `red_team_0_7_8.rs` | 7 | **keep + fix slowness** | 6 HostEval-ScalarFn-F1 locks are 0.01s each. The 1 `#[ignore]`'d `lint_subtree_invocation_...` test scanned the whole repo twice (~86-100s); scoped down to a representative `crates docs` subtree, which still proves the PR #93 path-canonicalization invariant (`lint .` vs explicit subtree walk produce identical `doc-filename-convention` error counts), and un-`#[ignore]`'d. |
| `phase3j_pre_std.rs` | 12 (3 active) | **keep, uncertain** | 9 already `#[ignore]`'d manual gates. 3 active tests (`..._distinct_seeds_differ`, `cross_function_seed_stdlib_kaiming_uniform_uses_handler_seed`, `cross_function_seed_stdlib_normal_like_advances_rng_per_random_op`) each do a full `chelis build` + gcc + run to verify RNG-seed plumbing determinism end-to-end. ~26s each. Genuine unique coverage; not duplicated elsewhere. Kept. |
| `phase_a_bundled_loader.rs` | 2 | **keep, uncertain** | Named architectural property oracle for the chelis-std bundle/loader. No `#[ignore]` by design. ~18s for a real `reef build`. Genuine unique coverage. |
| `cli.rs::phase3a_reef_std_acceptance_oracle` | 1 of 205 | **keep, uncertain** | ~26s reef-std acceptance oracle. Genuine unique coverage. |

## What changed

- **New file** `crates/chelis-cli/tests/production_stdlib_typechecks.rs`: one
  `#[test]` per unique production stdlib `.ch` file (19 files), each checked
  exactly once. nextest parallelizes the per-file tests across the global pool,
  so wall cost is one stdlib-check deep, not 19 serial. This replaces the 16 +
  3 scattered `production_stdlib_*` tests in `wsc_v3` and
  `wsc_stdlib_generalization`, plus the 2 redundant `check_stdlib_*` tests in
  `wsa8` and the 2 redundant production-file tests in `rt3`.
- `wsc_v3_stdlib_finish.rs`: Section 5 (16 tests + `stdlib_path` helper)
  deleted. Sections 1-4 retained.
- `wsc_stdlib_generalization.rs`: the 3 `production_stdlib_*_sig_file_type_checks`
  tests + `stdlib_path` helper deleted. All synthetic stub-sig tests retained.
- `rt3_adversarial.rs`: `production_stdlib_linear_now_builds_clean` and
  `production_stdlib_attention_typechecks_clean` + `stdlib_path` helper
  deleted. All synthetic adversarial tests retained.
- `wsa8_monomorphization_build.rs`: `check_stdlib_linear_still_clean_post_wsa8`
  and `check_stdlib_attention_still_clean_post_wsa8` deleted. Build-path tests
  retained.
- `red_team_0_7_8.rs`: `lint_subtree_invocation_matches_dot_for_doc_filename_convention`
  scoped to `crates docs` instead of the whole repo, and un-`#[ignore]`'d. The
  invariant (the two invocations produce equal `doc-filename-convention` error
  counts) is unchanged; only the corpus walked is smaller. `docs/manual_gates.md`
  entry for this test removed since it is back on the default gate.

## Coverage preserved

Every production stdlib file that was checked pre-trim is still checked exactly
once in `production_stdlib_typechecks.rs`. Every synthetic type-system /
adversarial test is retained. Every shipped-fix regression lock named in the
trim brief still runs. The only behavior removed is *re-running the same
production-stdlib check N times across N files*.
