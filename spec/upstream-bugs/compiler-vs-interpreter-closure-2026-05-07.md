# Compiler-vs-Interpreter Discrepancy Closure (v0.6.1)

**Status:** mostly closed (1 deferred)
**Filed:** 2026-05-07
**Owning phase:** Closure campaign (cross-phase)

## Context

The v0.6.1 catalog of compiler-vs-interpreter discrepancies (front-end /
IR evaluator / C backend) hit during the porting work was closed in a
single multi-agent campaign. The foundational principle: **a program
that passes `chelis check` should run in both `chelis test` (IR
evaluator) and `chelis build --target c` (C backend), or fail in
`chelis check` with a clear message.**

The closure was dispatched as 5 parallel worktree-isolated agents on
disjoint surfaces, plus 2 follow-up agents (Bucket 2 narrow tail and
the Phase 2 parity harness). Below is the closure column for each
catalog item.

## Closure column

| Bucket | Catalog item | Status | Commit(s) | Regression test |
|---|---|---|---|---|
| 1 | `grad` not in host runtime | **CLOSED** | 412fa61 | `cli::eval_grad_inline_form_in_host_runtime`, `cli::eval_grad_locally_bound_form_in_host_runtime`, `cli::eval_grad_wrapper_fn_param_form_in_host_runtime` |
| 1 | `realize` not in host runtime | **CLOSED** | 412fa61 | `cli::eval_realize_is_identity_in_host_runtime` |
| 1 | `vmap` not in host runtime | **CLOSED** | 412fa61 | `cli::eval_vmap_in_host_runtime` |
| 2 | C backend rejects inline `grad(f)(x)` | NO-OP (not actually rejected post-investigation) | — | n/a — Agent E confirmed inline form already lowers cleanly |
| 2 | C backend rejects locally-bound `let g = grad(f); g(x)` | **CLOSED** | d9996eb | `cli::build_c_grad_locally_bound_alias_form_lowers`, `cli::build_c_grad_locally_bound_alias_form_matches_inline_form_output` |
| 2 | Wrapper-fn-param refactor breaks `chelis test` (root-count divergence) | **CLOSED** | 412fa61 | covered by `cli::eval_grad_wrapper_fn_param_form_in_host_runtime` (Agent A's IR-evaluator path obviates the divergence) |
| 3 | `relu`/`sigmoid` not in host runtime | **CLOSED** | 375bd4f | `cli::bucket3_relu_runs_in_eval_and_c_lanes`, `cli::bucket3_sigmoid_runs_in_eval_and_c_lanes` |
| 3 | `gelu`/`silu`/`tanh` not in host runtime (or C) | **CLOSED** | 375bd4f | `cli::bucket3_gelu_tanh_approx_runs_in_eval_and_c_lanes`, `cli::bucket3_silu_runs_in_eval_and_c_lanes`, `cli::bucket3_tanh_runs_in_eval_and_c_lanes` |
| 4a | `expand` shape divergence (typer says rank+1, evaluator says rank) | **CLOSED** | 34ea538 | `chelis_compiler_api::runtime::tests::host_runtime_expand_singleton_input_inserts_not_replicates`, `cli::build_c_linreg_expand_singleton_bias_keeps_rank2_shape` |
| 4b | `to_tensor` rejects 2D Python-style literals | **CLOSED** | ad52c64 | `host_runtime_to_tensor_accepts_2d_float_literal`, `host_runtime_to_tensor_accepts_3d_float_literal`, `host_runtime_to_tensor_rejects_ragged_2d_literal`, `cli::build_c_to_tensor_2d_nested_literal_matches_eval_output` |
| 4c | Polymorphic dim leakage (`d36` undeclared) | **CLOSED** | 6b91d7b | `chelis_ir::dag::tests::symbolic_occurrences_sibling_sweep_picks_up_const_dims`, `chelis_ir::dag::tests::symbolic_occurrences_panics_when_dim_has_no_load_source`, `cli::build_c_polymorphic_top_level_tensor_dims_are_declared` |
| 4d | HOF f32 wrappers fail in C codegen | **CLOSED** | bf5313a | `cli::build_c_higher_order_scalar_fn_param_emits_wrapper` |
| 4e | Pipe operator drops shape on tensor activation chains | **CLOSED** | 5bdb85f, 08a7c44 | `cli::build_c_pipe_into_user_defined_unary_tensor_fn_matches_nested_call` |
| 5 | `with seed(...)` rejected by C backend (project-wide blocker) | **CLOSED** | d22d4a0 + follow-up fix | `cli::build_c_with_seed_uniform_like_succeeds`, `cli::build_c_with_seed_is_deterministic_across_runs`, `cli::build_c_with_seed_no_longer_blocks_sibling_build`, `phase3j_pre_oracle_build_path_repros_uniform_like_seed_succeeds`, `phase3j_pre_oracle_build_path_repros_uniform_like_seed_distinct_seeds_differ` |
| 5 | `with seed(...)` over function calls (`with seed { kaiming_uniform(...) }` where stdlib calls `uniform_like`) | **CLOSED (C host path)** | follow-up fix | `cli::cross_function_seed_local_wrapper_uses_handler_seed_in_c_backend`, `phase3j_pre_std::cross_function_seed_stdlib_kaiming_uniform_uses_handler_seed`, `phase3j_pre_std::cross_function_seed_stdlib_normal_like_advances_rng_per_random_op` |
| 6a | `cast(t, bf16)` rejected | **DEFERRED** | n/a | bf16 end-to-end is genuine multi-day cross-backend work (types + IR evaluator + C/HIP/Metal codegen + spec + examples). Not attempted in this campaign |
| 6b | `chelis check` is single-file only | **CLOSED** | d2cb0ed | `cli::check_directory_walks_ch_files_and_aggregates_json`, `cli::check_empty_directory_emits_empty_files_array`, `cli::check_single_file_keeps_legacy_report_shape` |
| 6c | `chelis test` requires repo-root cwd | **CLOSED** | d2cb0ed | `phase3t_test_smoke::chelis_test_resolves_reef_root_from_target_file_path`, `phase3t_test_smoke::chelis_test_errors_clearly_when_no_reef_anywhere` |

## Phase 2 — three-runtime parity harness

`crates/chelis-cli/tests/parity.rs` (commit db28694) walks the
executable corpus in `examples/` and asserts that `chelis check` /
`chelis eval` / `chelis build --target c` + binary all agree on output
to `1e-6` relative tolerance for tensor outputs (byte-equal otherwise).

Result on landing: 13 active tests passed, 2 were ignored:
- `parity_transformer_block_library_only` — environmental (`cblas.h`
  missing)
- `parity_mnist_library_only` — caught a real C-backend codegen bug:
  emits `-(__arg0_*)` on a non-numeric arg in `loss`. **Closure
  follow-up filed; not part of the original catalog.**

Follow-up result: `parity_mnist_library_only` is active again. The C
host type inference and tensor-helper routing now keep `softmax |> log
|> mul |> sum |> neg |> mean` on tensor values, so generated `mnist.c`
does not emit unsupported `sum`/`mean` comments or unary minus on a
non-numeric temporary.

## Out-of-scope items (per catalog)

- `chelis surf` decompile round-trip — best-effort by design; `.dp`
  drift check is the load-bearing equivalence guarantee.
- Reef install requiring `GITHUB_TOKEN` for private shells — pre-launch
  repo visibility constraint, not a compiler bug.

## Follow-up work

1. **Bucket 6a — bf16 end-to-end.** Multi-day cross-backend
   implementation. Not attempted; tracked as separate work.
2. **Bucket 5 — cross-function seed plumbing.** Closed for the C host
   path by moving random-handler state into the generated host runtime.
   A `with seed(...)` scope now activates per-handler RNG state, and
   nested stdlib/user functions that call `uniform_like` draw from that
   active handler instead of falling back to baked seed `0`.
3. **MNIST C codegen bug** (parity-harness finding). Closed by keeping
   tensor reductions and `neg` in the tensor-helper path for the MNIST
   loss tail and preserving the `tensor[f32]` rank-0 result shape in
   generated C; `parity_mnist_library_only` is no longer ignored.

## Campaign mechanics

- 5 parallel agents on isolated worktrees (Buckets 1, 3, 4, 5, 2+6).
- Disk pressure hit 97% with concurrent worktrees; mitigated by
  cleaning each agent's `target/` after completion.
- 2 follow-up agents (Bucket 2 narrow tail + Phase 2 parity harness)
  ran after the main campaign.
- Conflict resolution at integration: cli.rs (additive) and lower.rs
  (`is_known_unary_builtin` predicate union with Bucket 4e's
  `unreachable!()` arm).
- Workspace gate `cargo test --workspace` green post-integration except
  for pre-existing `chelis-python::compile_and_load_job_links_runtime_into_shared_library`
  (env-dependent — `cblas.h` missing).
