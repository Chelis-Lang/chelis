# Chelis Manual Gates

`CLAUDE.md` requires every manual acceptance gate to have a documented command,
expected success condition, and owning phase, and that ignored tests "clearly mirror a
documented manual gate." This file is the canonical inventory.

If a test is `#[ignore]`'d, it must appear here with its full command and prerequisite.
If a manual gate is named in a spec doc but not yet wired as an `#[ignore]`'d test, it
appears in the **Not yet wired** section at the bottom.

Expected outcome for every row below: the named test (or script) exits 0 with no
assertion failures unless an explicit different success condition is given.

## Ignored tests (wired manual gates)

| Test | Crate | Manual command | Prerequisite | Owning phase |
|---|---|---|---|---|
| `g1_add_consts_gpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g1_add_consts_gpu -- --ignored --test-threads=1` | HIP-capable GPU + `hipcc` / `hiprtc` (Fedora/ROCm available locally per CLAUDE.md "Local HIP Environment") | 1a |
| `g2_neg_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g2_neg_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g2_exp_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g2_exp_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g2_log_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g2_log_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g2_sin_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g2_sin_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g2_sqrt_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g2_sqrt_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g3_add_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g3_add_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g3_mul_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g3_mul_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g3_max_elem_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g3_max_elem_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g3_cmplt_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g3_cmplt_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g4_sum_reduction_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_sum_reduction_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a / 1d |
| `g4_max_reduce_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_max_reduce_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a / 1d |
| `g4_symbolic_row_sum_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_symbolic_row_sum_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g4_symbolic_softmax_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_symbolic_softmax_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g4_symbolic_matmul_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_symbolic_matmul_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g4_symbolic_matmul_gpu_reuses_one_artifact_for_multiple_batch_sizes` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g4_symbolic_matmul_gpu_reuses_one_artifact_for_multiple_batch_sizes -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g5_expand_add_stride_zero` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g5_expand_add_stride_zero -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g6_permute_then_add` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g6_permute_then_add -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g7_host_device_roundtrip` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g7_host_device_roundtrip -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1c |
| `g8_multi_kernel_chain` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g8_multi_kernel_chain -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a / 1b |
| `g9_load_mapping` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g9_load_mapping -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1a |
| `g10_realize_materializes_view_on_gpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g10_realize_materializes_view_on_gpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1c |
| `g11_repeated_load_alias_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g11_repeated_load_alias_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1c |
| `g12_reused_slot_respects_logical_size` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g12_reused_slot_respects_logical_size -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1c |
| `gf1_fused_add_neg_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness gf1_fused_add_neg_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1b |
| `gf2_fused_three_way_chain_gpu_matches_cpu` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness gf2_fused_three_way_chain_gpu_matches_cpu -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1b |
| `g13_tiny_segmented_sum_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g13_tiny_segmented_sum_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g13_small_segmented_max_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g13_small_segmented_max_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g13_large_segmented_sum_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g13_large_segmented_sum_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g14_staged_scalar_sum_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g14_staged_scalar_sum_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `g15_hipblas_matmul_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g15_hipblas_matmul_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` + hipBLAS | 1d |
| `g15_noncontiguous_matmul_fallback_matches_eval` | `chelis-backend-hip` | `cargo test -p chelis-backend-hip --test gpu_correctness g15_noncontiguous_matmul_fallback_matches_eval -- --ignored --test-threads=1` | HIP GPU + `hipcc` | 1d |
| `mnist_subset_full_pipeline` | `chelis-e2e` | `cargo test -p chelis-e2e --test mnist mnist_subset_full_pipeline -- --ignored` | MNIST data on disk; default `data/mnist/` or set `MNIST_DIR`; expected `final_acc > 0.50` on subset | 0h |
| `mnist_real_over_90_percent` | `chelis-e2e` | `cargo test -p chelis-e2e --test mnist mnist_real_over_90_percent -- --ignored` | MNIST data on disk; default `data/mnist/` or set `MNIST_DIR`; expected `final_acc > 0.90` on full data | 0h |
| `bench_phase1e_all_emits_structured_json_for_real_scope` | `chelis-e2e` | `cargo test -p chelis-e2e --test bench_phase1e -- --ignored` | Real benchmark scope: HIP GPU + PyTorch in `py/.venv` for cross-backend comparisons | 1e |
| `pipeline_transformer_model_lowers` | `chelis-e2e` | `cargo test -p chelis-e2e --test pipeline pipeline_transformer_model_lowers -- --ignored` | None beyond default toolchain (gated as long-running transformer compile) | 1f / 2d |
| `phase3b_python_manual_acceptance_oracle` | `chelis-python` | `cargo test -p chelis-python --test manual_phase3b -- --ignored` | `py/.venv` populated with PyTorch; `bindings/python` installed via `uv pip install` (the test installs it) | 3b |
| `phase3bii_python_manual_acceptance_oracle` | `chelis-python` | `cargo test -p chelis-python --test manual_phase3bii -- --ignored` | `py/.venv` with torch/numpy; `bindings/python` installed via `uv pip install` (the test installs it) | 3b-ii |
| `build_hip_runs_scalar_string_foundation_and_matches_eval_output` | `chelis-cli` | `cargo test -p chelis-cli --test cli build_hip_runs_scalar_string_foundation_and_matches_eval_output -- --ignored` | HIP-capable GPU + `hipcc`; expected: compiled HIP binary stdout matches `chelis eval --file` stdout | 3c |
| `phase3m_rust_runtime_hip_manual_gate` | `chelis-cli` | `cargo test -p chelis-cli --test cli phase3m_rust_runtime_hip_manual_gate -- --ignored` | HIP-capable GPU + `hipcc`; expected: build emits `libchelis_runtime.a` + `chelis_runtime.h` + `chelis_hip_runtime.h`, no `chelis_runtime.c`, compiled binary matches eval stdout | 3m |
| `pseudo_nautilus_parity_script_runs_with_scipy` | `chelis-cli` | `cargo test -p chelis-cli --test phase3t_pseudo_nautilus pseudo_nautilus_parity_script_runs_with_scipy -- --ignored` | `python3` with scipy installed; expected: parity script exits 0, prints parity table with max `\|diff\|` below 1e-5 | 3t |
| `worker_stack_overflow_in_one_file_does_not_kill_sibling_file` | `chelis-cli` | `cargo test -p chelis-cli --test phase3t_subprocess_isolation worker_stack_overflow_in_one_file_does_not_kill_sibling_file -- --ignored` | None beyond default toolchain; gated because the recursion depth needed to overflow the 32 MB worker stack is implementation-dependent. Expected: `tests/crash.ch` reports `FAIL (worker exited ...)` or `FAIL (worker killed by signal ...)`; sibling `tests/fine.ch::test_ok` reports `PASS`; parent exits 1. The sibling regression in default CI is `worker_crash_in_one_file_does_not_kill_sibling_file` (no `--ignored`), which uses the `CHELIS_TEST_FORCE_ABORT` env-var hatch on `cmd_internal_test_file` — included by `cargo test --workspace`. | 3t |

## Not yet wired (named in spec, no `#[ignore]`'d test)

| Manual gate | Documented command | Prerequisite | Owning phase |
|---|---|---|---|
| Phase 2a effect-surface manual check | `cargo run -q -p chelis-cli -- check $tmpdir/unhandled_random.ch` then `... handled_random.ch` (full script in `chelis_phase2_plan.md` §2a Acceptance Gate) | None beyond default toolchain; expected: first check reports `UnhandledEffect`/`Random`, second reports no effect errors | 2a |
| Phase 2b linearity manual check | Hand-written program allocating two large tensors, consuming one to produce another, verifying compiler accepts and runtime does not double-free | None beyond default toolchain | 2b |
| Phase 2c macro manual check | Define a custom `attention(q, k, v)` macro in Surf; expand it; verify the Deep matches the documented expansion | None beyond default toolchain | 2c |
| Phase 2e MCP manual check | Connect Claude (or another MCP-capable agent) to `chelis tide mcp`; have agent perform documented compiler tasks | MCP-capable agent client | 2e |
| Phase 2f editor host gate | Open a `.ch` file in VS Code with the Chelis extension; observe live diagnostics, hover, Deep toggle, fitness score | VS Code (or Cursor / Windsurf) + Chelis extension installed | 2f |
| Phase 2g `chelis cove` user gate | A user (not the developer) runs `cargo run -p chelis-cli -- cove --file examples/mnist.ch`; sees Deep, edits Surf, observes live fitness/diagnostics, triggers compile/eval inside the TUI | None beyond default toolchain | 2g |
| Phase 3j-pre release tag gate | Steps in `phase3j_pre_release.md` §Release Mechanism — `git tag -a vX.Y.Z`, `git push origin vX.Y.Z`, then `gh release view vX.Y.Z` shows tarball + checksum | GitHub push permission on `chelis-lang/chelis` | 3j-pre |
| Phase 3j Nautilus cross-repo CI | Downstream `chelis-lang/nautilus` `main` CI green, producing the published `v0.1.0` release (commit `20c5553`) | Cross-repo CI access | 3j |
| Phase 3k Coral GPU manual gate | "GPU: numeric column operations compile to HIP and produce correct results (manual gate)" — concrete command not yet documented | HIP-capable GPU + `hipcc` | 3k |
| `phase3k_coral_oracle` (named oracle, not implemented) | `cargo test -p chelis-cli phase3k_coral_oracle -- --exact` | Implementation pending in 3k coding work | 3k |
| `phase3l_shoals_oracle` (named oracle, not implemented) | `cargo test -p chelis-cli phase3l_shoals_oracle -- --exact` | Implementation pending in 3l coding work | 3l |
| `phase3n_octant_oracle` (named oracle, not implemented) | `cargo test -p chelis-cli phase3n_octant_oracle -- --exact` | Implementation pending in 3n coding work | 3n |
| `phase3o_octant_oracle` (named oracle, not implemented) | `cargo test -p chelis-cli phase3o_octant_oracle -- --exact` | Implementation pending in 3o coding work | 3o |
| `phase3j_pre_std_oracle` (named, differs from shipped) | `cargo test -p chelis-cli phase3j_pre_std_oracle -- --exact` (shipped suite is `cargo test -p chelis-cli --test phase3j_pre_std`) | Reconcile naming or accept shipped suite as the de facto oracle | 3j-pre |
| Phase 3f `skill_suite.rs` SKILL.md v2 validation | Run `skill_suite.rs` (when implemented) over every SKILL.md example against the current compiler | Implementation pending; depends on full Phase 3 surface | 3f |
| Phase 3t chelis-std self-test corpus | `cargo test -p chelis-cli --test phase3t_chelis_std_self -- --ignored --nocapture` | None beyond default toolchain. Expected: 1 passed (the harness), reporting `>= 150` chelis-std self-tests; current is 185. Runtime ~170s, off the 60s inner-loop budget per CLAUDE.md — invoked manually or in a long-form CI job. | 3t |
| #39 captured type-checker gap (defsig dim leak) | `cargo test -p chelis-cli --test phase3t_typechecker_gaps -- --ignored --nocapture` | Currently FAILS by design: pins the bug where a function declared with concrete tensor dims whose body produces wildcard dims silently accepts mismatched-shape arguments. Re-enable + flip to default once the ascription-narrowing fix lands. | 3t |

## Also See

- [`phase_oracles.md`](phase_oracles.md) — every numbered phase with its acceptance oracle command
- [`/CLAUDE.md`](../CLAUDE.md) — the agent contract that makes this index mandatory; see "Manual Gates" section
- [`spec/design/phase1a_kernel_codegen.md`](../spec/design/phase1a_kernel_codegen.md) — owning HIP gate spec
- [`spec/design/chelis_phase2_plan.md`](../spec/design/chelis_phase2_plan.md) — Phase 2 manual gate descriptions
- [`spec/design/chelis_phase3_plan.md`](../spec/design/chelis_phase3_plan.md) — Phase 3 manual gate descriptions
