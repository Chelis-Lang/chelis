# Chelis Phase Oracles

`AGENTS.md` / `CLAUDE.md` requires that **every phase name one authoritative completion oracle**
("a single command, a named suite, or a documented manual runner"). This file is the
canonical index of those oracles. If a phase row is missing, that phase has not yet
identified a completion oracle and should be treated as a gap, not an implicit pass.

Status legend:

- **default gate** — runs in the default-feature full workspace nightly. PRs run
  all lib/bin unit targets and only the integrations in `.config/ci-test-targets.toml`.
  A default-gate phase claim requires its full oracle on the candidate.
- **continuous gate** — runs through `scripts/gate.py` in hosted CI and the
  documented local pre-push subset, but is not a workspace test binary
- **nightly gate** — runs in **Linux Extended Validation** (`heavy-e2e.yml`),
  daily at 03:17 UTC or by manual dispatch. Its combined workspace/dtype pass
  includes all non-ignored default-feature tests, including those also run on PRs.
- **manual gate** — requires `#[ignore]` plus a documented prerequisite; see
  [`manual_gates.md`](manual_gates.md)
- **dedicated nightly gate** — runs as its own extended-validation worker; its
  result feeds the nightly failure report, not a required PR status context
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
| 0g | `cargo test -p chelis-ir --lib grad` (reverse-mode AD on DAG; analytical-vs-finite-difference unit suite in `src/grad.rs`) | `spec/design/chelis_project_plan.md` §0g | default gate |
| 0h | `cargo run --release -p chelis-e2e --bin train_mnist -- --epochs 5 --min-acc 0.90` (final test acc 0.9272 on the checked-in path); supporting MNIST e2e tests are listed in `docs/manual_gates.md` | `spec/design/chelis_project_plan.md` §0h | manual gate (long-running release runner and MNIST e2e tests; default `cargo test --workspace` does **not** run this milestone) |
| 0i | `cargo test -p chelis-cli` (Tide v0.1: `tide`, `deep`, `surf`, `fmt`, `eval`) | `spec/design/chelis_project_plan.md` §0i | default gate |

## Phase 1

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 1a | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1a_kernel_codegen.md` §Acceptance Oracle | manual gate (HIP hardware; locally runnable per [`local_hip_environment.md`](local_hip_environment.md)) |
| 1b | `cargo test --workspace` (plus rerun the 1a HIP manual oracle) | `spec/design/phase1b_fusion.md` §Acceptance Oracle | default gate (with HIP rerun as supplement) |
| 1c | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1c_memory_planning.md` §Acceptance Oracle | manual gate (HIP hardware) |
| 1d | `cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/phase1d_flattening.md` §Acceptance Oracle | manual gate (HIP hardware) |
| 1f | `cargo test -p chelis-e2e --test example_corpus_validate` | `spec/design/phase1f_executable_grammar.md` §Authoritative oracle | default gate |
| 1 (symbolic dims carry-forward) | `cargo test --workspace` plus 1a HIP gate | `spec/design/phase1_symbolic_dims.md` | default gate + manual gate |

## Phase 2

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 2a | `cargo test -p chelis-effects` + `cargo test -p chelis-types --test linearity` + `cargo test -p chelis-cli --test cli check_reports_a_keyless_dropout_as_an_arity_error` + `cargo test -p chelis-compiler-api --test resource_target_admission` + `cargo test -p chelis-cli --test issue_735_device_fence` | `spec/design/chelis_phase2_plan.md` §2a Acceptance Gate | default gate |
| 2b | `cargo test -p chelis-types --test linearity` | `spec/design/chelis_phase2_plan.md` §2b Acceptance Gate | default gate |
| 2c | `cargo test -p chelis-macros --test expansion` | `spec/design/chelis_phase2_plan.md` §2c Acceptance Gate | default gate |
| 2d | `cargo test -p chelis-ir --test vmap` + `cargo test -p chelis-e2e --test spec_suite` + `cargo test -p chelis-e2e --test pipeline` | `spec/design/chelis_phase2_plan.md` §2d Acceptance Gate | default gate |
| 2e | `cargo test -p chelis-tide --test api` + `cargo test -p chelis-tide --test mcp` | `spec/design/chelis_phase2_plan.md` §2e Acceptance Gate | default gate |
| 2f | `cargo test -p chelis-lsp` plus a documented manual editor gate (open `.ch` in VS Code; observe live diagnostics, hover, Deep toggle) | `spec/design/chelis_phase2_plan.md` §2f Acceptance Gate | default gate (library) + manual gate (editor host) |
| 2g | Manual: `cargo run -p chelis-cli -- cove --file examples/mnist.ch`; user (not developer) confirms live Deep + diagnostics + compile/eval inside the TUI | `spec/design/chelis_phase2_plan.md` §2g Acceptance Gate | manual gate |
| 2s | Deferred to Phase 4a (per phase2 plan); no Phase 2 oracle | `spec/design/chelis_phase2_plan.md` §Deferred: Seed Corpus | aspirational (deferred) |

## Phase 3

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| 3a | `cargo test -p chelis-cli --test cli phase3a_reef_std_acceptance_oracle` (heavy: publishes chelis-std and builds a dependent app) | `spec/design/chelis_project_plan.md` §3a | nightly gate (Linux Extended Validation) |
| 3b | `cargo test -p chelis-python --test manual_phase3b -- --ignored` | `spec/design/chelis_project_plan.md` §3b | manual gate (PyTorch venv) |
| 3b-ii | `cargo test -p chelis-python --test manual_phase3bii -- --ignored` | `spec/design/chelis_project_plan.md` §3b-ii | manual gate (torch/numpy venv) |
| 3c | `cargo test -p chelis-cli --test cli phase3c_scalar_string_acceptance_oracle` | `spec/design/chelis_phase3_plan.md` §3c Acceptance Oracle | default gate |
| 3d | Corpus oracle: `examples/list_foundation.ch`, `examples/dict_foundation.ch`, `examples/iter_foundation.ch` survive `chelis fmt`/`check`/`eval`/`build --target c` and compiled output matches eval. Exercised through `cargo test --workspace` example-corpus tests | `spec/design/chelis_phase3_plan.md` §3d Acceptance Oracle | default gate |
| 3e | `cargo test -p chelis-cli --test cli phase3e_pipe_first_acceptance_oracle` | `spec/design/chelis_project_plan.md` §3e | default gate |
| 3f | `skill_suite.rs` validation of every SKILL.md example against the current compiler, plus eval lift on target base models | `spec/design/chelis_phase3_plan.md` §3f Acceptance Oracle | aspirational (post-rest-of-Phase-3) |
| 3g | `cargo test -p chelis-cli --test std_io_pipeline io_pipeline_acceptance_oracle -- --ignored --exact --nocapture` | `spec/design/chelis_phase3_plan.md` §3g Acceptance Oracle | manual gate (build/link/run the CSV and JSON example; full Std.Io/Csv/Json/Parquet ignored suite is tracked separately in `docs/manual_gates.md`) |
| 3h | `cargo test -p chelis-cli --test cli phase3h_numeric_acceptance_oracle -- --ignored --exact --nocapture` | `spec/design/chelis_project_plan.md` §3h | manual gate (numeric build/link oracle) |
| 3i | `cargo test -p chelis-cli --test std_package_acceptance -- --ignored --nocapture` | `spec/design/chelis_phase3_plan.md` §3i Acceptance Oracle | manual gate (package typing; Std.Time and Std.Decimal callables are fenced by #2779 and #2778) |
| 3j-pre | `cargo test -p chelis-cli --test production_stdlib_typechecks` + `cargo test -p chelis-cli --test std_package_acceptance` (in-repo std surface: `Std.Embedding`; `Std.Time` and `Std.Decimal` callables are fenced by #2779 and #2778; School provides the neural-network, loss, optimizer, and initializer libraries) plus ignored batch suites in `docs/manual_gates.md` | `spec/design/chelis_project_plan.md` §3j-pre + `phase3j_pre_release.md` | manual gate (heavyweight eval/import/check/build acceptance) |
| 3j | Downstream Nautilus `main` CI green producing the published `v0.1.0` release (commit `20c5553`); the placeholder `phase3j_nautilus_oracle` was retired | `spec/design/chelis_phase3_plan.md` §3j Acceptance Oracle | manual gate (cross-repo CI) |
| 3k | `cargo test -p chelis-cli --test coral_prerequisites -- --ignored --nocapture` | `spec/design/chelis_phase3_plan.md` §3k Acceptance Oracle | manual gate (Coral prerequisite/regression acceptance); downstream Coral-owned oracle still future work |
| 3l | `cargo test -p chelis-cli --test shoals_oracle phase3l_shoals_oracle -- --ignored --exact --nocapture` | `spec/design/chelis_phase3_plan.md` §3l Acceptance Oracle | manual gate (Shoals finance oracle; ~5 minutes locally). Focused grad-property lower/type-check smoke: `cargo test -p chelis-cli --test shoals_oracle phase3l_shoals_oracle_grad_greeks_match_analytic -- --ignored --exact --nocapture` |
| 3m | `cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture` | `spec/design/phase3m_rust_runtime_rewrite.md` §Acceptance Oracle | default gate (HIP variant `phase3m_rust_runtime_hip_manual_gate` is `#[ignore]`d) |
| 3n | `cargo test -p chelis-cli phase3n_octant_oracle -- --exact` | `spec/design/phase3n_octant.md` §1.5 Acceptance oracle | aspirational (named, not yet implemented) |
| 3o | `cargo test -p chelis-cli phase3o_octant_oracle -- --exact` | `spec/design/phase3n_octant.md` §2.5 Acceptance oracle | aspirational (named, not yet implemented) |
| 3t | `chelis test tests/` exits 0 on the migrated Nautilus and Coral test suites; `parity/run_parity.py` continues to pass for the scipy/pandas comparison subset (in-repo: `test_command_smoke.rs` covers the default smoke path; exhaustive Std.Test, Decimal fence, build-path, and pseudo-Nautilus suites are manual gates) | `spec/design/chelis_phase3_plan.md` §3t Acceptance Oracle + `chelis_native_testing_plan.md` | default gate (smoke) + manual gates (exhaustive/std/pseudo suites) + aspirational (downstream test migration) |

## Cross-Phase Closure Campaigns

| Campaign | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| Runtime representation hardening · Phase 0 | `uv run --managed-python --python 3.11 --no-project python scripts/runtime_representation_oracle.py --phase 0` | `spec/design/runtime_representation.md` §Phase 0 | delivered and directly runnable; inherited by the later composites |
| Runtime representation hardening · Phase 1 | `uv run --managed-python --python 3.11 --no-project python scripts/runtime_representation_oracle.py --phase 1` | `spec/design/runtime_representation.md` §Phase 1 | delivered and inherited by Phase 2; Phase 0 inventory/mutations plus frozen host/C execution receipts |
| Runtime representation hardening · Phase 2 | `uv run --managed-python --python 3.11 --no-project python scripts/runtime_representation_oracle.py --phase 2` | `spec/design/runtime_representation.md` §Phase 2 | dedicated nightly gate under the stable `runtime-representation-phase0-oracle` job identity; runs Phase 1 first, then frozen generated-ABI, metadata-plan, HIP owner, Python/DLPack, hermetic HIP-header census, and Metal enrollment receipts; needs `clang` on PATH |
| Runtime bundle identity (#1354) | `.venv/bin/python scripts/runtime_bundle_oracle.py` from a clean committed head; inspect `target/runtime-bundle/receipt.json` for every mandatory row and the tested SHA | `spec/08-backends.md` §2.1 and `spec/11-ffi.md` §2 | manual exact-head integration gate (CLI/Python native calls, mutations, sealed Nix packages, wheel, and source-free relocation); PR `script-unit` runs only its stdlib contract tests. HIP/Metal device execution remains separate in [`manual_gates.md`](manual_gates.md); rerun this full command on final merged `main` before closure. |
| Integer dtype spelling migration | `.venv/bin/python scripts/integer_dtype_spelling_oracle.py` (final line `INTEGER DTYPE SPELLING ORACLE: PASS`) | `spec/design/integer_dtype_spelling.md` | delivered and locally runnable; canonical Surf/Deep, explicit v0.18 migrations, corpus closure, and unchanged external interchange vocabulary |
| Deep substrate handover | `cargo test -p chelis-compiler-api --test deep_authoring` + `cargo test -p chelis-tide --test mcp replace_function_body` + `cargo test -p chelis-tide --test mcp add_function` + `cargo test -p chelis-tide --test api replace_function_body` + `cargo test -p chelis-tide --test api add_function` + `cargo test -p chelis-types duplicate_defsig` + `cargo test -p chelis-validate duplicate_defsig` + `cargo test -p chelis-cli --test surf_round_trip` | `spec/design/chelis_agent_editing_surface.md` | default gate |
| Deep authoring L2 query/cascade + `.dp` SMT parity | `cargo test -p chelis-deep --test authoring` + `cargo test -p chelis-compiler-api --test deep_authoring` + `cargo test -p chelis-tide --test mcp deep_query_and_rename_tools_are_model_facing_contracts` + `cargo test -p chelis-tide --test api deep_query_and_rename_http_endpoints_lock_preimage_contract` + `cargo test -p chelis-prove --features smt property_runner::tests::f7_deep -- --nocapture` + `cargo test -p chelis-tide --features smt --test mcp deep_user_property_proves_at_smt_tier_through_tide -- --nocapture` | `spec/design/chelis_agent_editing_surface.md` + `spec/design/chelis_deep_authoring_handover.md` | default gate plus SMT feature gate |
| Compiled value ownership Phase 0 | `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 0` (final line `COMPILED VALUE OWNERSHIP PHASE 0: PASS`) | `spec/design/compiled_value_ownership.md` §Phase 0 | delivered and locally runnable; CI has advanced the stable `compiled-value-ownership-phase0-oracle` job identity to the inherited Phase 2 gate |
| Compiled value ownership Phase 1 | `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 1` (final line `COMPILED VALUE OWNERSHIP PHASE 1: PASS`) | `spec/design/compiled_value_ownership.md` §Phase 1 | delivered and inherited by Phase 2; exact ownership-ledger execution receipts cover the heap-kind, Option-node, mapped-file, and guarded-write suites |
| Compiled value ownership Phase 2 | `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2` (final line `COMPILED VALUE OWNERSHIP PHASE 2: PASS`) plus `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase launch` (final line `COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS`) | `spec/design/compiled_value_ownership.md` §Phase 2 | dedicated nightly gate under the stable `compiled-value-ownership-phase0-oracle` job identity; launch subset is #1362 Tier 1 item A |
| Compiled value ownership Phase 3 | `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 3 --require-hip` (final line `COMPILED VALUE OWNERSHIP PHASE 3: PASS`) | `spec/design/compiled_value_ownership.md` §Phase 3 | hardware acceptance pending under #1286/#1214, separate from implementation delivery; manual `ownership-hip.yml` workflow on a configured AMD runner, with setup and receipt instructions in `docs/local_hip_environment.md`; absent from default PR CI |

## Phase A (Reef Distribution Unblock)

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| A · Item 6 (`--from-github`) | `cargo test -p chelis-cli --test reef_install_from_github phaseA_item6_from_github_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 6 | default gate |
| A · Item 7 (`--bootstrap`) | `cargo test -p chelis-cli --test reef_install_bootstrap phaseA_item7_bootstrap_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 7 | default gate |
| A · Item 8 (auto-fetch during build) | `cargo test -p chelis-cli --test reef_build_autofetch phaseA_item8_autofetch_build_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 8 | default gate |
| A · Item 9 (lockfile remote-origin) | `cargo test -p chelis-cli --test reef_lockfile_remote_origin phaseA_item9_lockfile_origin_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 9 | default gate |
| A · Bundled chelis-std loader | `cargo test -p chelis-cli --test bundled_chelis_std_loader phaseA_bundled_chelis_std_loader_property_oracle -- --exact` | `spec/design/reef_distribution.md` §Item 7 (bundling) + `spec/design/chelis_canonical_reference.md` §5.4 | nightly gate (Linux Extended Validation) |
| A · Real-network end-to-end (Item 6) | `GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli --test reef_install_from_github phaseA_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 6 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 7) | `GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli --test reef_install_bootstrap phaseA_item7_real_bootstrap_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 7 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 8) | `GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli --test reef_build_autofetch phaseA_item8_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 8 | manual gate (real GitHub + `GITHUB_TOKEN`) |
| A · Real-network end-to-end (Item 9) | `GITHUB_TOKEN=$(gh auth token) cargo test -p chelis-cli --test reef_lockfile_remote_origin phaseA_item9_real_github_manual_gate -- --ignored --exact` | `spec/design/reef_distribution.md` §Item 9 | manual gate (real GitHub + `GITHUB_TOKEN`) |

## Phase D (Differentiable Programming)

Committed scope per `spec/design/differentiable_language.md` and the
`spec/12-roadmap.md` §Differentiable programming track. D0 is the spec
lock; oracles for D1–D6 are named here as aspirational until the owning
agent dispatch picks the executable test fixture for each phase.

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| D0 | `grep -F "spec/design/differentiable_language.md" spec/12-roadmap.md` returns at least one hit (proves the canonical doc was landed and cross-referenced from the roadmap) | `spec/design/differentiable_language.md` §Phase 0 | default gate (doc grep) |
| D1 | Numerical-agreement suite covering AD through `if`, `match`, `while`, `for`, and a recursive function — exact named test TBD when D1 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 1 | aspirational |
| D2 | Field-wise gradient suite covering struct/record gradients, ADT-tagged match gradients, and higher-order-function gradients — exact named test TBD when D2 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 2 | aspirational |
| D3 | Effect-aware AD suite: pathwise (Normal/Uniform/Beta), REINFORCE (Categorical/Bernoulli), `raises`/`state` composition, plus the small-Bayesian VI convergence test — exact named test TBD when D3 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 3 | aspirational |
| D4 | Implicit-differentiation suite: `fix` agreement vs unrolled baseline, analytical-`argmin` quadratic, KKT-derived constrained-optimum gradient — exact named test TBD when D4 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 4 | aspirational |
| D5 | Differentiability-typing suite: positive/negative annotation checks, `chelis check --show-inferred` regression, `Lipschitz(K)` property verifier — exact named test TBD when D5 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 5 | aspirational |
| D6 | Corpus oracle: every program in `examples/differentiable/` survives `chelis fmt`/`check`/`eval`/`build` and the on-ramp primer's worked snippets match their checked-in outputs — exact named test TBD when D6 dispatch picks the fixture | `spec/design/differentiable_language.md` §Phase 6 | aspirational |

## Phase H (Hydronnx — ONNX shell)

Committed scope per `spec/design/hydronnx.md` and the
`spec/12-roadmap.md` §Hydronnx track. `Hydronnx` is the Chelis shell;
`ONNX` is the upstream interchange format. H0 is the spec lock;
oracles for H1–H5 are named here as aspirational until the owning
agent dispatch picks the executable test fixture for each phase.

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| H0 | `grep -F "spec/design/hydronnx.md" spec/12-roadmap.md` returns at least one hit (proves the canonical doc was landed and cross-referenced from the roadmap) | `spec/design/hydronnx.md` §Phase 0 | default gate (doc grep) |
| H1 | Parser-inventory agreement suite: a small ONNX model (MobileNet-class or linear regression) loaded through hydronnx produces the same inventory as `onnx.checker.check_model` in Python, and malformed inputs surface the documented diagnostics — exact named test TBD when H1 dispatch picks the fixture | `spec/design/hydronnx.md` §Phase 1 | aspirational |
| H2 | Per-operator numerical-agreement suite against ONNX Runtime across the v0.1 core operator subset, plus one end-to-end model per strong-fit category — exact named test TBD when H2 dispatch picks the fixture | `spec/design/hydronnx.md` §Phase 2 | aspirational |
| H3 | End-to-end loading suite: weights load, `load_model` produces a callable Chelis function with correct outputs, `inspect_model` matches actual model structure, `load_model_with_opts` overrides take effect, documented failure cases (custom op, unsupported opset, corrupted weights) surface the documented errors — exact named test TBD when H3 dispatch picks the fixture | `spec/design/hydronnx.md` §Phase 3 | aspirational |
| H4 | Type-discipline integration suite: dimension types on loaded signatures, wrong-shape call-site rejection, property attachment + verification, AD composition (positive and negative — non-differentiable operator surfaces a pinned diagnostic), composition with hand-written Chelis preprocessing/post-processing — exact named test TBD when H4 dispatch picks the fixture | `spec/design/hydronnx.md` §Phase 4 | aspirational |
| H5 | Corpus oracle: every program in the hydronnx examples directory survives `chelis fmt`/`check`/`eval`/`build`, the migration-guide snippets compile, and the worked image-classification / object-detection / tabular examples produce the documented outputs — exact named test TBD when H5 dispatch picks the fixture | `spec/design/hydronnx.md` §Phase 5 | aspirational |

## Phase K (Kerrent — GPU kernel authorship)

Committed scope per `spec/design/kerrent.md` and the
`spec/12-roadmap.md` §Kerrent track. `Kerrent` is the Chelis language
feature for authoring GPU kernels in Chelis source; `Triton` is the
upstream kernel compiler Kerrent emits to. K0 is the spec lock;
oracles for K1–K6 are named here as aspirational until the owning
agent dispatch picks the executable test fixture for each milestone.
Addendums KA–KF (per `spec/design/chelis_project_plan.md` §Kerrent
Track) are post-v1 extensions and do not appear here.

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| K0 | `grep -F "spec/design/kerrent.md" spec/12-roadmap.md` returns at least one hit (proves the canonical doc was landed and cross-referenced from the roadmap) | `spec/design/kerrent.md` (the spec lock is the doc's existence; the body enumerates Milestones 1–6 only, so K0 has no in-spec milestone anchor) | default gate (doc grep) |
| K1 | Parser-correctness suite for the `kernel` annotation and the v1 tile-level primitives (`tile.load`, `tile.store`, `tile.dot`, `tile.reduce`, `tile.mask`, …); kernel-annotated functions get a distinct AST representation distinguishable from tensor-level functions — exact named test TBD when K1 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 1 | aspirational |
| K2 | Kernel-IR-layer type-checking suite: tile-level operations are first-class IR nodes; dimension types compose through tile scope; mismatched-shape kernel bodies fail at type-check — exact named test TBD when K2 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 2 | aspirational |
| K3 | Triton IR emission suite: each tile-level operation lowers to a defined Triton IR equivalent and the output validates against Triton's IR specification — exact named test TBD when K3 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 3 | aspirational |
| K4 | Build-integration suite: kernel-annotated Chelis programs compile through `chelis build`; PTX/AMDGCN artifacts are produced and link cleanly; the Triton dependency is handled by the build system — exact named test TBD when K4 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 4 | aspirational |
| K5 | Runtime-integration suite: kernel calls from tensor-level Chelis launch correctly against Triton artifacts; memory layout matching at the kernel boundary is correct; cross-vendor execution (NVIDIA via PTX, AMD via AMDGCN) produces matching results — exact named test TBD when K5 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 5 | aspirational |
| K6 | FlashAttention proof point: a FlashAttention-shaped fused attention kernel written in Kerrent replaces the current attention decomposition; transformer inference achieves competitive performance vs PyTorch+CUDA for the attention block, with numerical agreement vs the existing decomposed path — exact named test TBD when K6 dispatch picks the fixture | `spec/design/kerrent.md` §Milestone 6 | aspirational |

## Phase M (Metal Backend)

| Phase | Oracle command | Owning spec doc | Status |
|---|---|---|---|
| M0 | `grep -F "[MTLDevice newLibraryWithSource:]" spec/design/chelis_metal_backend_plan.md` returns at least one hit (proves §3.3 was rewritten away from the metal-rs Rust runtime to the string-emission-only model) | `spec/design/chelis_metal_backend_plan.md` §3.3 | default gate (doc grep) |
| M1 | `cargo build --workspace` + `cargo test -p chelis-cli --test cli -- target_metal` + `cargo tree -p chelis-cli` no-Apple-SDK-deps guard | `spec/design/chelis_metal_backend_plan.md` §9 (M1) | default gate |
| M2 | `cargo test -p chelis-backend-metal --test codegen_structure` | `spec/design/chelis_metal_backend_plan.md` §9 (M2) | default gate |
| M3 | `.github/workflows/macos-nightly.yml` `macos-workspace-shard` job runs `python3 .github/scripts/smoke_macos_metal.py` on shard 2 and exits 0; the stable `macos-smoke` aggregate requires both workspace shards | `spec/design/chelis_metal_backend_plan.md` §9 (M3) | nightly 04:17 UTC or manual dispatch; not a default PR gate |
| M4 | `cargo test -p chelis-backend-metal --test codegen_structure -- reduction` | `spec/design/chelis_metal_backend_plan.md` §9 (M4) | default gate |
| M5 | `cargo test -p chelis-backend-metal --test codegen_structure -- matmul_tiled` | `spec/design/chelis_metal_backend_plan.md` §9 (M5) | default gate |
| M6 | `cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1` | `spec/design/chelis_metal_backend_plan.md` §9 (M6) | manual gate (Apple Silicon Mac with Metal device available) |
| M7 | `cargo test -p chelis-backend-metal --test codegen_adversarial` | `spec/design/chelis_metal_backend_plan.md` §9 (M7) | default gate |

## Also See

- [`manual_gates.md`](manual_gates.md) — every `#[ignore]`'d test with its manual command and prerequisite
- [`/AGENTS.md`](../AGENTS.md) / [`/CLAUDE.md`](../CLAUDE.md) — the agent contract that makes this index mandatory
- [`local_hip_environment.md`](local_hip_environment.md) — local ROCm/HIP runbook for HIP manual gates
- [`local_macos_environment.md`](local_macos_environment.md) — macOS first-exec (syspolicyd) wedge runbook; preflight probe `scripts/preflight_exec_probe.py`
- [`spec/design/chelis_project_plan.md`](../spec/design/chelis_project_plan.md) — top-level phase ledger
- [`spec/design/chelis_phase3_plan.md`](../spec/design/chelis_phase3_plan.md) — detailed Phase 3 implementation plan
- [`spec/design/chelis_metal_backend_plan.md`](../spec/design/chelis_metal_backend_plan.md) — Phase M Metal backend
