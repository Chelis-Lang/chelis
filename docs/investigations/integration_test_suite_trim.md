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

## Heavy e2e split (per-PR gate vs nightly)

Follow-up to the trim above. The trim deduplicated *redundant* heavy
invocations; this step removes the *remaining* heavy invocations from the
per-PR gate entirely, because even one ~28s test sets a hard wall-clock floor
nextest cannot parallelize away. The heavy e2e suite now runs in a separate
nightly workflow (`.github/workflows/heavy-e2e.yml`, `nightly` nextest
profile); the per-PR `ci`/`default` profiles exclude exactly that set.

### Heavy e2e selection (currently ON the per-PR pass, >8s wall)

Measured via `cargo nextest run --workspace --profile ci` per-test timing on a
32-core box.

| Test | Wall | Path | Verdict |
|---|---|---|---|
| `phase3j_pre_std::cross_function_seed_stdlib_normal_like_advances_rng_per_random_op` | ~28s | reef build + gcc + run; `normal_like` RNG-per-op advance | **move to nightly** |
| `phase3j_pre_std::cross_function_seed_stdlib_kaiming_uniform_uses_handler_seed` | ~28s | reef build + gcc + run; `kaiming_uniform` threads handler seed, asserts generated C | **move to nightly** |
| `phase3j_pre_std::phase3j_pre_oracle_build_path_repros_uniform_like_seed_distinct_seeds_differ` | ~28s | reef build + gcc + run; `uniform_like` byte-exact stdout for distinct seeds | **move to nightly** |
| `cli::phase3a_reef_std_acceptance_oracle` | ~27s | reef-std acceptance oracle | **move to nightly** |
| `phase_a_bundled_loader::phaseA_bundled_chelis_std_loader_property_oracle` | ~19s | real `reef build`; bundled chelis-std loader property oracle | **move to nightly** |
| `cli::reef_check_accepts_sig_only_shell_imports` | ~16s | `reef check`, sig-only shell imports | **move to nightly** |
| `wsa8_monomorphization_build::build_stdlib_linear_succeeds` | ~14s | `chelis build` stdlib `linear.ch` + gcc | **move to nightly** |
| `wsa8_monomorphization_build::build_stdlib_attention_succeeds` | ~14s | `chelis build` stdlib `attention.ch` + gcc | **move to nightly** |
| `phase3t_reef_install::reef_install_from_monorepo_populates_registry_and_unblocks_check` | ~10s | `reef install` from monorepo | **move to nightly** |
| `cli::reef_build_emits_shell_and_archive` | ~10s | `reef build` shell + archive | **move to nightly** |
| `production_stdlib_typechecks::*` (19 tests) | ~9-10s each | full-stdlib `chelis check`, one per stdlib `.ch` file | **move to nightly** |

The `wsa8_monomorphization_build` binary is moved whole (its 4 fast negative
`build_rejects_*` tests ride along; keeping the binary together is cleaner than
splitting it, and they are part of the same monomorphization build-acceptance
surface).

### Dedupe verdict

The split brief also asked for an aggressive dedupe of the heavy set down to a
small high-signal suite. The categorization above was done with that lens. The
verdict: **every heavy test currently on the per-PR pass moves to nightly
intact; none were deleted in this change.** Reasoning, per the
"do not guess-delete a load-bearing e2e invariant" constraint:

- `production_stdlib_typechecks::*` (19 tests) is *already* the deduped result
  of the trim above. The trim audit's explicit invariant is "every production
  stdlib file that was checked pre-trim is still checked exactly once."
  Collapsing those 19 into one representative test would re-drop coverage a
  prior audit deliberately preserved, and would lose per-file failure
  attribution. Re-deduping it is an orchestrator decision, not a safe
  mechanical collapse.
- The three `phase3j_pre_std` RNG-seed tests each pin a *distinct* assertion:
  byte-exact `uniform_like` output for distinct seeds; `kaiming_uniform`
  threading the handler seed (verified against generated C, not just stdout);
  `normal_like` advancing the RNG per random op. They are related but not
  duplicates. Collapsing them needs a judgment call on which assertions are
  load-bearing.
- `cli::phase3a_reef_std_acceptance_oracle` and
  `phase_a_bundled_loader::phaseA_bundled_chelis_std_loader_property_oracle`
  are *named architectural property oracles* (see `phase_oracles.md` /
  `manual_gates.md`). Deleting a named oracle is out of scope for a CI-shape
  change.
- `wsa8_monomorphization_build` build tests and `production_stdlib_typechecks`
  exercise *different paths* (`chelis build` + gcc + codegen vs `chelis
  check`), so they are not duplicates of each other.

### Dedupe candidates flagged for orchestrator review

Not acted on here; each needs a load-bearing-invariant judgment call:

- `production_stdlib_typechecks::*`: 19 tests, identical operation
  (`chelis check <stdlib file>`), each re-typechecks the whole transitive
  stdlib graph. Candidate: keep one representative deepest-import-graph file
  and drop the rest, accepting the loss of per-file attribution. Counter-
  argument: the trim audit preserved per-file coverage on purpose.
- `phase3j_pre_std` three RNG-seed tests: candidate to collapse to one
  end-to-end RNG-seed-determinism test if the distinct assertions
  (byte-exact output, handler-seed-in-C, per-op advance) can be folded into a
  single fixture without losing any of them.
- `wsa8_monomorphization_build` `build_stdlib_linear_succeeds` vs
  `build_stdlib_attention_succeeds`: candidate to keep one representative
  stdlib build if monomorphization coverage does not actually differ between
  `linear.ch` and `attention.ch`.

The dedupe examples named in the split brief that target *manual-gate*
(`#[ignore]`'d) tests rather than per-PR tests -- the gelu/rmsnorm numeric
reference across `phase3j_pre_std` + `phase3j_pre_std_batch3` +
`phase3t_build_runtime_gaps`, `phase3j_pre_oracle_integrated_build_c` vs the
per-item `_build_path_repros_*` tests, and the `phase3j_pre_std_batch2/3/3b/4`
files -- are out of scope for this change: those tests are already 100%
`#[ignore]`'d, so they are already off the per-PR gate. Deduping the manual-gate
set is a separate workstream.
