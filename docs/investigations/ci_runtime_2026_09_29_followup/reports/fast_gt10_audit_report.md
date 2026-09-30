# Fast Tests (Linux): >10 s test audit, September 8–29, 2026

**Scope.** Audited current `origin/main` `5e2e8630db704a0e9ac64d384ce39d16f763ad2e` using `git show`. A bounded date/runner-stratified sample of **15 successful PR Fast runs, September 17–29** (14 hosted, one warm) contains **80,336 JUnit cases, 327 executions >10 s, 42 distinct identities**; Fast jobs and `ci-fast/coverage.json` passed. Post-merge `main` run `36585051950` at `465a3623be19bd68ada6c39aa7552b63e0ccf8c3` was inspected separately, excluded from counts. September 13 run `34735211733` had expired artifacts; 14-day retention blocks September 8–14 per-test analysis. Fast selection began around September 11. This sample is not a period estimate. Inherited PR/run/criticality totals (`target/pr-ci-3week-evidence.md`).

**Receipts.** Run `gh run download RUN_ID -R Chelis-Lang/chelis -n junit-linux-fast -D target/fast-gt10-audit/RUN_ID/junit-linux-fast`, then repeat with `-n ci-fast-receipts -D target/fast-gt10-audit/RUN_ID/ci-fast-receipts`. Parse `junit.xml` and `ci-fast/{timing,coverage}.json`. `target/fast-gt10-audit/RUN_ID/{jobs,artifacts}.json` records runner, job steps and artifact IDs; `analysis.json` and the temporary Python scripts preserve results. JUnit includes nested CLI/C subprocesses. Appendix medians condition on observations >10 s.

| PR CI run | Exact head SHA | Runner | Fast job ID | Job/Gate s | >10 cases |
|---:|---|---|---:|---:|---:|
| 35233483425 | `01093cd9b6f9944a5208ec899f2b86f761cbd43f` | hosted | 105243606051 | 1095/1008 | 24 |
| 35453706585 | `1f41645e2002ce0013a951bbc28791e7949d2dbc` | hosted | 105925275698 | 1051/918 | 24 |
| 35637666070 | `bf1f8217a5efceb9de0a68d287d147e72e78f4b0` | hosted | 106459433561 | 995/875 | 18 |
| 35700978350 | `b0828e9eca2413643f330dcbe6bce2ad8e6830ae` | hosted | 106659033807 | 1186/1085 | 28 |
| 35968940824 | `f9d8bd5064d875a84eaf8390882c3018bbe4b968` | warm | 107534162935 | 526/467 | 15 |
| 36179203178 | `3641250dfdac4fce2fce2868af30c9c34f4bd37d` | hosted | 108217698715 | 1129/880 | 18 |
| 36299900356 | `a589b98ba474c4f020f6d3aa534f0f9ecdba871f` | hosted | 108565648229 | 1261/1111 | 26 |
| 36463542957 | `fe68aa1fcdfe485f4f1e8d0919638bb64065cf2f` | hosted | 109068467177 | 1510/1388 | 25 |
| 36469331313 | `a573b60133cdde34141136438d6caee835527118` | hosted | 109088077515 | 1802/1647 | 25 |
| 36509012547 | `b67246adfce660cfa3148ba8ac2769b2f92db22e` | hosted | 109217091774 | 1895/1740 | 25 |
| 36581803867 | `4dab9a2afef18f614cea636e05c47e1816cf83f8` | hosted | 109452432433 | 1319/1129 | 19 |
| 36594166425 | `20f726163a2b0aa942725f6f347da98d78b43b1f` | hosted | 109495217150 | 1536/1376 | 25 |
| 36596085022 | `4bfcf238c3fed0e0c9577d911d39b0d8ee438ab7` | hosted | 109501718579 | 1235/1051 | 18 |
| 36598074659 | `16ae3630f07d8f1d7fa566d2be808f5bbb833869` | hosted | 109508506256 | 985/798 | 13 |
| 36601246959 | `a76110a6fcdb9603ba7c5a7a5882ad2a6ec93ae6` | hosted | 109519400629 | 1624/1411 | 24 |

## Ranked costs and guarantees

1. **Source guard.** `A::source_arch::guarded_upper_consumers_do_not_recreate_the_semantic_pipeline`: **15/15** PRs >10 s; median **210.2**, max **475.7 s**. This lib unit scans five roots for duplicate semantic pipelines; planted controls test that scope (`crates/chelis-compiler-api/src/source_arch.rs:2712-2801,4781-4791`). Its proof excludes other roots; directory-read errors yield no files. Four post-#2754 PRs measured 307.7/228.0/160.4/313.0 s, without a paired head. Profile parse, inventory and propagation; retain the PR scan.
2. **Claimed joins.** The former standing 100-cell case measured **204.3–350.8 s** (median **261.3**) on four PRs. Current Fast runs three regression tests, including a six-cell canary at **12.7–22.7 s** on four later PRs; `issue_2413_claimed_join_lanes_full` keeps all 100 (`crates/chelis-cli/tests/issue_2413_claimed_join_lanes.rs:1-47`; `.../issue_2413_claimed_join_lanes_full.rs:1-59`; `.config/ci-test-targets.toml:519-526`). The canary pins prior wrong empty values via successes, traps and typed refusals in host/DAG Eval and C. Direct edits trigger change-owned CI; expansion/nightly own the full matrix. **Other cells lack unconditional PR execution**.
3. **Standing CLI tests.** `C::runtime_extent_slice_b` has recurring **37–55 s** grad cases with good values, typed traps and Eval/C parity (`crates/chelis-cli/tests/runtime_extent_slice_b.rs:9445-9614`). `C::issue_1739_diagonal_runtime_bound::no_shipped_example_gains_a_host_lane_guard` has median **30.7**, max **46.3 s**: it checks each executable example's generated guards and good/bad controls (`crates/chelis-cli/tests/issue_1739_diagonal_runtime_bound.rs:1-46,742-816,1820-1874`). Both are standing (`.config/ci-test-targets.toml:791-823`).
4. **Fixture costs.** `C::issue_1650_empty_tensor_cli` builds C for **9 dtypes × nesting × emptiness** (median **34.6 s**), with wrong/unresolved dtype negatives (`crates/chelis-cli/tests/issue_1650_empty_tensor_cli.rs:74-108,183-253`). Keyed dropout checks Eval/C bits against a separate key oracle (`.../dropout_fixed_stream_cli.rs:320-358,563-604`). `A::compiled_context_authenticated_handoff` repeatedly makes a real-sized stdlib fixture; its ~30 s fixed-point positive has payload, digest and foreign-library negatives (`crates/chelis-compiler-api/tests/compiled_context_authenticated_handoff.rs:1-20,159-191,282-329,379-425,492-527`). `chelis-ir` units test scaling (`crates/chelis-ir/src/host.rs:22056-22095`).

**Critical path.** Hosted run `36509012547`: Fast job **1,895 s**, Gate **1,740**, principal Nextest group **1,313**. Source guard occupied **476 s (27% of Gate)** and old join **351 s (20%)**; the guard ended ~69 s before the final JUnit case. Post-merge PR `36601246959`: Gate **1,411 s**, group **977**, guard **313 s (22%)**, ending ~73 s before the last case (`target/fast-gt10-audit/RUN_ID/{jobs.json,ci-fast-receipts/ci-fast/timing.json,junit-linux-fast/junit.xml}`). Run `36509012547` has a >10 s JUnit sum of **1,389 s**, exceeding group wall time because cases overlap. These are **occupancy, not savings**; other work remains. No paired #2753/#2754 trial.

## Recommendations

- **Batch exact preparations:** time check/Eval/C build/link/run separately. Share compilation only for identical source, caller, toolchain and target; execute each good/trap observation in its own process with its own assertion. Batch the 36 empty-tensor variants into fewer C translation units, keeping all observations and negatives. Keep diagonal per-example attribution. Share immutable handoff setup only with independent foreign/corruption controls. Savings are unmeasured.
- **Keep the PR claim explicit:** source, extent and example guards retain their current PR assertions. Nightly breadth plus a canary loses PR detection outside it. Full workspace ownership is `.github/workflows/heavy-e2e.yml:3-5,80-86`; final expansion is conditional/manual (`docs/ci_validation.md:269-318`). Six appendix identities are retired; the old join identity moved to `_full`.
- **Reshard with receipts:** `scripts/ci_test_targets.py:53-99,106-145,187-260` selects every default-feature workspace lib/bin unit and standing integration. CLI crate splitting leaves test selection intact. Parallel Fast shards duplicate setup/build and need disjoint fail-closed receipts. Unifying proofs hides failures. Compare same-head hosted finish and runner work.

**Limits/status.** No large build or tracked edits; final `git status --porcelain=v1` was empty. Early artifacts expired; subphase times are unavailable. Temporary work is under `target/fast-gt10-audit/`.

## Appendix: all 42 sampled PR identities observed >10 s

`A = chelis-compiler-api`, `C = chelis-cli`, `I = chelis-ir`; expand prefixes for exact identities. `n` and median/max use only >10 s PR observations. The old join identity moved; six other names are retired.


**`A::builtin_named_kernel_inputs`**

- `builtin_spelled_data_parameters_agree_in_eval_and_native_c` — 4; 12.2/19.1

**`A::compiled_context_authenticated_handoff`**

- `cached_program_is_a_checker_fixed_point` — 8; 29.6/32.0
- `authenticated_decode_rejects_a_type_environment_from_another_library` — 8; 20.5/21.5
- `authenticated_decode_rejects_a_digest_minted_for_other_bytes` — 7; 18.6/20.3
- `both_decode_routes_reconstruct_identical_contexts` — 5; 13.3/13.8
- `authenticated_decode_rejects_a_payload_rewritten_after_encode` — 5; 10.9/11.2

**`A::fixed_control_c`**

- `native_mask_threshold_uses_arithmetic_width_and_strict_less_than` — 7; 19.8/22.6
- `native_special_words_reject_mask_sign_and_nan_corruption` — 7; 17.4/18.8
- `sealed_native_dropout_matches_evaluator_and_restarts_each_public_invocation` — 1; 13.5/13.5
- `native_source_ad_replay_finalizes_nonbinary_rate_division` — 5; 11.3/11.5
- `native_source_ad_replays_signed_and_nonfinite_stored_words` — 1; 10.4/10.4

**`A::fixed_control_host_c`**

- `context::context_native_source_order_saved_mask_and_shapes_survive_decode` — 3; 11.8/12.3

**`A::local_ascription_activation`**

- `a_vmapped_arms_local_ascription_checks_only_when_a_row_takes_the_arm` — 7; 16.8/17.1

**`A::runtime`**

- `tests::issue_1829_interpreter_entry_bounds_kernel_decision_probes` — 3; 11.7/20.1
- `shared_values::tests::deep_values_compare_on_a_small_stack` — 7; 14.6/15.9

**`A::source_arch`**

- `guarded_upper_consumers_do_not_recreate_the_semantic_pipeline` — 15; 210.2/475.7

**`C::dropout_fixed_stream_cli`**

- `a_keyed_dropout_under_a_runtime_arm_reads_only_its_own_key_in_eval_and_c` — 8; 34.7/36.6
- `a_dropout_in_an_unselected_runtime_arm_takes_no_ordinal_in_eval_or_c` — 3; 20.7/32.6

**`C::issue_1418_recursive_cast_targets`**

- `published_arange_and_linspace_match_eval_at_each_requested_width` — 7; 23.0/26.8

**`C::issue_1582_vmap_std_scalar`**

- `imported_min_and_max_work_under_vmap_in_eval_and_native_c` — 5; 16.6/18.7
- `unimported_min_and_max_are_unbound_variables` — 1; 10.7/10.7

**`C::issue_1650_empty_tensor_cli`**

- `generated_c_empty_tensor_metadata_matches_every_checked_dtype` — 15; 34.6/38.0

**`C::issue_1739_diagonal_runtime_bound`**

- `no_shipped_example_gains_a_host_lane_guard` — 15; 30.7/46.3
- `c_selected_match_result_claims` — 15; 22.2/27.2
- `value_aliases_keep_the_guard_at_the_original_producer` — 13; 14.8/16.3

**`C::issue_1956_named_axis_inputs_cli`**

- `initializer_failure_preserves_original_error_and_channels` — 5; 13.0/14.4
- `dead_capture_keeps_default_style_admission_and_no_initializer` — 4; 11.3/12.0
- `formal_shadow_keeps_default_style_admission_and_no_initializer` — 4; 11.2/12.0

**`C::issue_2205_container_last_use`**

- `alias_controls_and_witnesses_keep_their_values_on_both_lanes` — 7; 12.8/13.3

**`C::issue_2205_dict_last_use`**

- `dict_alias_controls_keep_their_values_on_both_lanes` — 7; 10.7/11.1

**`C::issue_2413_claimed_join_lanes`**

- `every_lane_returns_the_taken_arm_or_its_claims_trap_at_a_claimed_join` — 4; 261.3/350.8
- `shared_origin_untaken_zero_claims_keep_the_taken_value` — 4; 19.4/22.7

**`C::runtime_extent_slice_b`**

- `independent_grad_entry_failures_are_ordered_on_eval_and_c` — 15; 43.2/55.2
- `independent_grad_entry_claims_agree_on_eval_and_c` — 15; 37.0/43.9
- `independent_grad_computed_claims_keep_producer_on_eval_and_c` — 15; 38.0/43.5
- `a_prim_scalar_wrt_is_still_silent_on_eval_and_refused_on_c` — 15; 27.9/30.9
- `an_aggregate_typed_wrt_is_still_lane_divergent` — 13; 17.1/21.6
- `issue_1932_mapped_grad_entry_witness_matrix_executes_exactly_on_both_lanes` — 14; 15.2/19.1
- `scalar_gradient_roots_keep_target_order_and_public_leaf_types` — 10; 11.4/13.2
- `malformed_parameter_ranks_keep_the_existing_helper_diagnostic` — 9; 11.0/11.8

**`I::host`**

- `tests::issue_1835_kernel_decision_work_is_linear` — 10; 14.9/21.5
- `tests::issue_1205_host_lowering_work_is_linear` — 1; 16.9/16.9
