## Phase 1e: Fixed-Workload Benchmarking and Reference Comparison

**Goal:** prove the shipped Phase 1 GPU backend on a small fixed workload set, record real numbers, and compare Chelis CPU, Chelis HIP, and PyTorch through the repo-local Python benchmark environment when it is prepared.

### Authoritative Oracle

```sh
cargo run --release -p chelis-e2e --bin bench_phase1e -- --model all --emit-json benchmarks/results/latest.json
```

This command is the Phase 1e completion oracle. It emits a single machine-readable report covering:

- `linreg` training
- `mnist` training + inference on a fixed subset
- `transformer` forward pass

If HIP, PyTorch, or the local MNIST IDX subset prerequisite are missing, the report must
record an explicit `skipped` status and reason. Silent fallback is not acceptable.

Local/manual prerequisite for real PyTorch numbers:

```sh
uv venv --python 3.12 py/.venv
uv pip install --python py/.venv/bin/python --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --prerelease allow torch torchaudio torchvision
```

CI does **not** install PyTorch. In CI, the oracle is still expected to emit structured
PyTorch `skipped` reports rather than fail or silently downgrade.

Default-test/manual-gate split:

- `cargo test --workspace` should only exercise the fast structural benchmark coverage
- the real fixed-scope benchmark integration test is `#[ignore]` so the default test loop
  does not pay the full multi-minute benchmark cost
- manual benchmark-test gate:

```sh
cargo test -p chelis-e2e --test bench_phase1e -- --ignored
```

### Benchmark Surface

Phase 1e is intentionally a **thin fixed runner**, not a general benchmarking framework.
The only public CLI surface is:

```sh
cargo run -p chelis-e2e --bin bench_phase1e -- --model <linreg|mnist|transformer|all> --emit-json <path>
```

No suite composition, tuning flags, arbitrary epochs, or backend selectors are exposed.
The benchmark configuration is fixed in code and documented in `benchmarks/RESULTS.md`.

### Fixed Workloads

| Model | Chelis example | Scope | Notes |
|---|---|---|---|
| Linear regression | `examples/linreg.ch` | train | synthetic dense regression with fixed data |
| MNIST MLP | `examples/mnist.ch` | train + inference | fixed subset, fixed seed/init; requires local MNIST IDX files for a real run |
| Transformer-block-style forward | `examples/transformer_block.ch` | forward only | sequence-only block within current rank-2 matmul support |

Transformer benchmark shapes:

- `seq_len = 128`
- `d_model = 256`
- `n_heads = 4`
- `head_dim = 64`
- `d_ff = 1024`

The benchmark remains within the currently shipped op surface.
**CNN / LeNet / max-pool are excluded from Phase 1e** because they are not yet part of the
implemented executable benchmark path.

### Comparison Policy

- Chelis CPU is the in-repo semantic oracle for the benchmark runner
- Chelis HIP is the Phase 1 target backend under test
- PyTorch is the local/manual comparison backend, but not the primary oracle
- the benchmark runner selects PyTorch via `CHELIS_BENCH_PYTHON` or the repo-local
  `py/.venv`; it does not fall back to an ambient global `python3`
- the benchmark runner removes stale shell `HSA_OVERRIDE_GFX_VERSION` values before invoking PyTorch so local ROCm probing reflects the actual hardware

Acceptance expectations:

- forward-only comparisons use direct tensor tolerances (`f32`, target `1e-5` to `1e-4`
  depending on the path)
- training trajectories are compared by **trend**, not by exact stepwise equality
- MNIST and linear-regression comparisons require loss to decrease on both sides
- MNIST comparison uses final-accuracy band agreement, not exact floating-point matching

### Deliverables

- `crates/chelis-e2e/src/bin/bench_phase1e.rs`
- benchmark support in `crates/chelis-e2e/src/bench.rs`
- executable examples in `examples/linreg.ch` and `examples/transformer_block.ch`
- PyTorch references under `benchmarks/pytorch/`
- checked-in report artifacts:
  - `benchmarks/results/latest.json`
  - `benchmarks/RESULTS.md`
- documented local Python benchmark environment in `py/.venv`, prepared manually with the
  ROCm gfx1151 nightly PyTorch install command above

Missing-MNIST behavior:

- `--model mnist` and `--model all` must still emit structured JSON when MNIST data is absent
- the `mnist` model report must become an explicit skip with the dataset reason, not a process abort

### Tests

- [x] `bench_phase1e` rejects invalid `--model`
- [x] fast structural smoke coverage emits valid benchmark JSON without requiring the full
  real benchmark scope
- [x] `bench_phase1e --model all` exercises the real fixed benchmark scope via the manual
  ignored-test gate
- [x] skipped backends always carry explicit reasons
- [x] missing MNIST data becomes a structured skip, not a CLI failure
- [x] missing benchmark Python interpreter becomes a structured PyTorch skip
- [x] new executable examples compile through the full Surf pipeline
- [x] benchmark integration coverage produces a report with CPU results and explicit backend statuses
