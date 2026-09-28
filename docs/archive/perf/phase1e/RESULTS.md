# Phase 1e Results

Historical capture from April 2026. The Phase 1e runner and PyTorch scripts have
been retired; the commands and setup below describe the original capture and do
not run on current `main`. The [JSON report](results.json) preserves its raw
measurements and skip reasons.

Original oracle:

```sh
cargo run --release -p chelis-e2e --bin bench_phase1e -- --model all --emit-json benchmarks/results/latest.json
```

Local PyTorch setup for real comparison runs:

```sh
uv venv --python 3.12 py/.venv
uv pip install --python py/.venv/bin/python --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --prerelease allow torch torchaudio torchvision
```

CI policy:

- `cargo test --workspace` does not install PyTorch.
- When the repo-local benchmark Python env is absent, Phase 1e emits explicit PyTorch `skipped` reports with a setup hint.
- The archived `results.json` is the manual local comparison artifact.

## Capture

- Date: `2026-04-07T21:11:52+01:00`
- Commit: `0654565`
- OS: `Linux 6.18.16-200.fc43.x86_64`
- CPU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`
- GPU: `Radeon 8060S Graphics (gfx1151)`
- HIP toolchain: `HIP 6.4.43484-9999`, clang 19
- PyTorch env: repo-local `py/.venv` via `uv`, Python `3.12`
- PyTorch version: `2.11.0a0+rocm7.11.0a20260106`
- MNIST data: local IDX files present in `data/mnist`

## Fixed workloads

- `linreg`: synthetic dense regression, batch `64`, `16` train batches, `4` test batches, `12` epochs
- `mnist`: checked-in MLP on a fixed subset, batch `32`, `32` train batches, `8` test batches, `5` epochs
- `transformer`: sequence-only transformer-block-style forward pass, `seq_len=128`, `d_model=256`, `n_heads=4`, `head_dim=64`, `d_ff=1024`, `5` iterations

## Summary

| Model | CPU run ms | HIP run ms | PyTorch run ms | HIP peak device bytes | Key result |
|---|---:|---:|---:|---:|---|
| `linreg` | 34.90 | 6765.68 | 68.41 | 50440 | All three backends agree; final loss `1.292e-5` |
| `mnist` | 21626.31 | 8344.74 | 386.21 | 14291116 | All three backends agree; fixed-run test accuracy `0.8242` |
| `transformer` | 3928.86 | 8109.79 | 92.56 | 280629248 | Forward outputs agree within `1.67e-6` max abs diff |

## Notes

- These timings measure the benchmarked train/forward execution loops after benchmark data loading, tensor allocation/setup, and parameter initialization; they are not isolated kernel timings.
- The PyTorch comparison scripts now run on ROCm GPU when `torch.cuda.is_available()` succeeds in the repo-local env.
- The benchmark runner strips the stale `HSA_OVERRIDE_GFX_VERSION=11.0.0` shell override before invoking PyTorch; without that cleanup this machine misreports as `gfx1100` and the ROCm PyTorch lane fails with an invalid kernel image.
- Timings vary noticeably between captures; `results.json` is the machine-readable artifact for the exact measured run.
- The HIP lane beats the generated CPU lane on the two training workloads in this capture.
- The transformer forward workload is still slower on HIP than on the generated CPU lane in this capture; Phase 1e accepts that because correctness and architecture validation are the primary gate.
- `results.json` is the machine-readable source of truth for the recorded metrics and explicit skip reasons.
