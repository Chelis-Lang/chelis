# Chelis Phase Oracles

`CLAUDE.md` requires that **every phase name one authoritative completion oracle**
("a single command, a named suite, or a documented manual runner"). This file is the
canonical index of those oracles. If a phase row is missing, that phase has not yet
identified a completion oracle and should be treated as a gap, not an implicit pass.

Status legend:

- **default gate** — runs as part of `cargo test --workspace` (the inner-loop suite)
- **manual gate** — requires `#[ignore]` plus a documented prerequisite; see
  [`manual_gates.md`](manual_gates.md)
- **aspirational** — oracle is named in the owning spec but not yet implemented as
  executable code; phase is not done until it exists

## Phase 0

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 0a | `cargo test --workspace` (project scaffold + CI green) | `spec/design/chelis_project_plan.md` §Phase 0 Snapshot | default gate |
| 0b | `cargo test -p chelis-deep` (Deep parser) | `spec/design/chelis_project_plan.md` §0b | default gate |
| 0c | `cargo test -p chelis-surf` (Surf parser + desugaring) | `spec/design/chelis_project_plan.md` §0c | default gate |
| 0d | `cargo test -p chelis-types` (HM inference, dims, precision) | `spec/design/chelis_project_plan.md` §0d | default gate |
| 0e | `cargo test -p chelis-ir` (RISC DAG construction) | `spec/design/chelis_project_plan.md` §0e | default gate |
| 0f | `cargo test -p chelis-backend-c` (C codegen + numerics) | `spec/design/chelis_project_plan.md` §0f | default gate |
| 0g | `cargo test -p chelis-ir --test grad` (reverse-mode AD on DAG) | `spec/design/chelis_project_plan.md` §0g | default gate |
| 0h | `cargo run --release -p chelis-e2e --bin train_mnist -- --epochs 5 --min-acc 0.90` (final test acc 0.9272 on the checked-in path) | `spec/design/chelis_project_plan.md` §0h | manual gate (long-running release runner; default `cargo test --workspace` does **not** run this milestone) |
| 0i | `cargo test -p chelis-cli` (Tide v0.1: `tide`, `deep`, `surf`, `fmt`, `eval`) | `spec/design/chelis_project_plan.md` §0i | default gate |

## Phase 1

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 1a | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1a_kernel_codegen.md` §Acceptance Oracle | manual gate (HIP hardware) |
| 1b | `cargo test --workspace` (plus rerun the 1a HIP manual oracle) | `spec/design/phase1b_fusion.md` §Acceptance Oracle | default gate (with HIP rerun as supplement) |
| 1c | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1c_memory_planning.md` §Acceptance Oracle | manual gate (HIP hardware) |
| 1d | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1d_flattening.md` §Acceptance Oracle | manual gate (HIP hardware) |
| 1e | `cargo test -p chelis-e2e --test bench_phase1e -- --ignored` (real benchmark scope; `bench_phase1e` smoke variants run by default) | `spec/design/phase1e_benchmarks.md` §Authoritative Oracle | manual gate (real scope); default gate covers smoke |
| 1f | `cargo test -p chelis-e2e --test phase1f_validate` | `spec/design/phase1f_executable_grammar.md` §Authoritative oracle | default gate |
| 1 (symbolic dims carry-forward) | `cargo test --workspace` plus 1a HIP gate | `spec/design/phase1_symbolic_dims.md` | default gate + manual gate |

## Phase 2

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 2a | `cargo test -p chelis-effects` + `cargo test -p chelis-types --test linearity` + `cargo test -p chelis-cli --test cli check_reports_unhandled_random_effect build_rejects_gpu_device_region_for_c_target` | `spec/design/chelis_phase2_plan.md` §2a Acceptance Gate | default gate |
| 2b | `cargo test -p chelis-types --test linearity` | `spec/design/chelis_phase2_plan.md` §2b Acceptance Gate | default gate |
| 2c | `cargo test -p chelis-macros --test expansion` | `spec/design/chelis_phase2_plan.md` §2c Acceptance Gate | default gate |
| 2d | `cargo test -p chelis-ir --test vmap` + `cargo test -p chelis-e2e --test spec_suite` + `cargo test -p chelis-e2e --test pipeline` | `spec/design/chelis_phase2_plan.md` §2d Acceptance Gate | default gate |
| 2e | `cargo test -p chelis-tide --test api` | `spec/design/chelis_phase2_plan.md` §2e Acceptance Gate | default gate |
| 2f | `cargo test -p chelis-lsp` plus a documented manual editor gate (open `.ch` in VS Code; observe live diagnostics, hover, Deep toggle) | `spec/design/chelis_phase2_plan.md` §2f Acceptance Gate | default gate (library) + manual gate (editor host) |
| 2g | Manual: `cargo run -p chelis-cli -- cove --file examples/mnist.ch`; user (not developer) confirms live Deep + diagnostics + compile/eval inside the TUI | `spec/design/chelis_phase2_plan.md` §2g Acceptance Gate | manual gate |
| 2s | Deferred to Phase 4a (per phase2 plan); no Phase 2 oracle | `spec/design/chelis_phase2_plan.md` §Deferred: Seed Corpus | aspirational (deferred) |

## Phase 3

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 3a | `cargo test -p chelis-cli --test cli phase3a_reef_std_acceptance_oracle` | `spec/design/chelis_project_plan.md` §3a | default gate |
| 3b | `cargo test -p chelis-python --test manual_phase3b -- --ignored` | `spec/design/chelis_project_plan.md` §3b | manual gate (PyTorch venv) |
| 3b-ii | `cargo test -p chelis-python --test manual_phase3bii -- --ignored` | `spec/design/chelis_project_plan.md` §3b-ii | manual gate (torch/numpy venv) |
| 3c | `cargo test -p chelis-cli --test cli phase3c_scalar_string_acceptance_oracle` | `spec/design/chelis_phase3_plan.md` §3c Acceptance Oracle | default gate |
| 3d | Corpus oracle: `examples/list_foundation.ch`, `examples/dict_foundation.ch`, `examples/iter_foundation.ch` survive `chelis fmt`/`check`/`eval`/`build --target c` and compiled output matches eval. Exercised through `cargo test --workspace` example-corpus tests | `spec/design/chelis_phase3_plan.md` §3d Acceptance Oracle | default gate |
| 3e | `cargo test -p chelis-cli --test cli phase3e_pipe_first_acceptance_oracle` | `spec/design/chelis_project_plan.md` §3e | default gate |
| 3f | `skill_suite.rs` validation of every SKILL.md example against the current compiler, plus eval lift on target base models | `spec/design/chelis_phase3_plan.md` §3f Acceptance Oracle | aspirational (post-rest-of-Phase-3) |
| 3g | `cargo test -p chelis-cli --test phase3g_io phase3g_text_pipeline_acceptance_oracle -- --nocapture` | `spec/design/chelis_phase3_plan.md` §3g Acceptance Oracle | default gate |
| 3h | `cargo test -p chelis-cli phase3h_numeric_acceptance_oracle -- --nocapture` | `spec/design/chelis_project_plan.md` §3h | default gate |
| 3i | `cargo test -p chelis-cli --test phase3i_std -- --nocapture` | `spec/design/chelis_phase3_plan.md` §3i Acceptance Oracle | default gate |
| 3j-pre | `cargo test -p chelis-cli phase3j_pre_std_oracle -- --exact` (named in plan; current shipped surface is `cargo test -p chelis-cli --test phase3j_pre_std`) | `spec/design/chelis_project_plan.md` §3j-pre + `phase3j_pre_release.md` | aspirational (named oracle differs from shipped suite name) |
| 3j | Downstream Nautilus `main` CI green producing the published `v0.1.0` release (commit `20c5553`); the placeholder `phase3j_nautilus_oracle` was retired | `spec/design/chelis_phase3_plan.md` §3j Acceptance Oracle | manual gate (cross-repo CI) |
| 3k | `cargo test -p chelis-cli phase3k_coral_oracle -- --exact` | `spec/design/chelis_phase3_plan.md` §3k Acceptance Oracle | aspirational (named, not yet implemented) |
| 3l | `cargo test -p chelis-cli phase3l_shoals_oracle -- --exact` | `spec/design/chelis_phase3_plan.md` §3l Acceptance Oracle | aspirational (named, not yet implemented) |
| 3m | `cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture` | `spec/design/phase3m_rust_runtime_rewrite.md` §Acceptance Oracle | default gate (HIP variant `phase3m_rust_runtime_hip_manual_gate` is `#[ignore]`d) |
| 3n | `cargo test -p chelis-cli phase3n_octant_oracle -- --exact` | `spec/design/phase3n_octant.md` §1.5 Acceptance oracle | aspirational (named, not yet implemented) |
| 3o | `cargo test -p chelis-cli phase3o_octant_oracle -- --exact` | `spec/design/phase3n_octant.md` §2.5 Acceptance oracle | aspirational (named, not yet implemented) |
| 3t | `chelis test tests/` exits 0 on the migrated Nautilus and Coral test suites; `parity/run_parity.py` continues to pass for the scipy/pandas comparison subset (in-repo: `phase3t_test_smoke.rs`, `phase3t_test_std.rs`, `phase3t_pseudo_nautilus.rs` cover the smoke + pseudo-nautilus pieces) | `spec/design/chelis_phase3_plan.md` §3t Acceptance Oracle + `chelis_native_testing_plan.md` | default gate (smoke) + aspirational (downstream test migration) + manual gate (scipy parity) |

## Phase A (Reef Distribution Unblock)

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| A · Item 6 (`--from-github`) | `cargo test -p chelis-cli --test phase_a_item6_from_github phase_a_item6_from_github_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 6 | default gate |
| A · Item 7 (`--bootstrap`) | `cargo test -p chelis-cli --test phase_a_item7_bootstrap phase_a_item7_bootstrap_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 7 | default gate |
| A · Item 8 (auto-fetch during build) | `cargo test -p chelis-cli --test phase_a_item8_autofetch_build phase_a_item8_autofetch_build_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 8 | default gate |
| A · Item 9 (lockfile remote-origin) | `cargo test -p chelis-cli --test phase_a_item9_lockfile_origin phase_a_item9_lockfile_origin_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 9 | default gate |
| A · Bundled chelis-std loader | `cargo test -p chelis-cli --test phase_a_bundled_loader phase_a_bundled_chelis_std_loader_property_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 7 (bundling) + `spec/design/chelis_canonical_reference.md` §5.4 | default gate |
| A · Real-network end-to-end (Item 6) | `cargo test -p chelis-cli phase_a_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 6 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 7) | `cargo test -p chelis-cli phase_a_item7_real_bootstrap_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 7 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 8) | `cargo test -p chelis-cli phase_a_item8_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 8 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 9) | `cargo test -p chelis-cli phase_a_item9_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 9 | manual gate (real GitHub + `GITHUB_TOKEN`) |

## Phase M (Metal Backend)

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| M0 | `grep -F "[MTLDevice newLibraryWithSource:]" spec/design/chelis_metal_backend_plan.md` returns at least one hit (proves §3.3 was rewritten away from the metal-rs Rust runtime to the string-emission-only model) | `spec/design/chelis_metal_backend_plan.md` §3.3 | default gate (doc grep) |
| M1 | `cargo build --workspace` + `cargo test -p chelis-cli --test cli -- target_metal` + `cargo tree -p chelis-cli` no-Apple-SDK-deps guard | `spec/design/chelis_metal_backend_plan.md` §9 (M1) | default gate |
| M2 | `cargo test -p chelis-backend-metal --test codegen_structure` | `spec/design/chelis_metal_backend_plan.md` §9 (M2) | default gate |
| M3 | `.github/workflows/ci.yml` `macos-smoke` job runs `python3 .github/scripts/smoke_macos_metal.py` and exits 0 | `spec/design/chelis_metal_backend_plan.md` §9 (M3) | default gate (macOS CI) |
| M4 | `cargo test -p chelis-backend-metal --test codegen_structure -- reduction` | `spec/design/chelis_metal_backend_plan.md` §9 (M4) | default gate |
| M5 | `cargo test -p chelis-backend-metal --test codegen_structure -- matmul_tiled` | `spec/design/chelis_metal_backend_plan.md` §9 (M5) | default gate |
| M6 | `cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/chelis_metal_backend_plan.md` §9 (M6) | manual gate (Apple Silicon Mac with Metal device available) |
| M7 | `cargo test -p chelis-backend-metal --test redteam_adversarial` | `spec/design/chelis_metal_backend_plan.md` §9 (M7) | default gate |

## Also See

- [`manual_gates.md`](manual_gates.md) — every `#[ignore]`'d test with its manual command and prerequisite
- [`/CLAUDE.md`](../CLAUDE.md) — the agent contract that makes this index mandatory
- [`spec/design/chelis_project_plan.md`](../spec/design/chelis_project_plan.md) — top-level phase ledger
- [`spec/design/chelis_phase3_plan.md`](../spec/design/chelis_phase3_plan.md) — detailed Phase 3 implementation plan
- [`spec/design/chelis_metal_backend_plan.md`](../spec/design/chelis_metal_backend_plan.md) — Phase M Metal backend
