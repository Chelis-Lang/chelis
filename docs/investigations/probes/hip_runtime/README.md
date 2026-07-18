# HIP runtime probes — #689 / #690 confirmation + #729 overflow-trap spike

Runtime evidence gathered on the gfx1151 workstation for issues that were
previously **emission-** or **inspection-proven only** (the audit machine, arm64
macOS, has no `hipcc`). Every artifact here was compiled and executed; the
measured outputs are attached to the owning issues (#689, #690, #736).

## Environment

- **Device**: AMD Radeon 8060S Graphics (`gfx1151`, Ryzen AI Max+ 395). Probe
  the GPU with `rocminfo` (source of truth), not `rocm-smi`.
- **Toolchain**: HIP 7.13.26162 / hipcc (AMD clang 23.0.0).
- **Compiler**: `chelis` 0.16.1 (built from this tree with `cargo build -p chelis-cli`).
- **HIP env is only valid through the wrapper.** Plain `cargo`/`hipcc`
  invocations inherit only the `environment.d/hip.conf` defaults and segfault
  hipBLAS-linked binaries at process exit (looks like a code bug, is purely
  environmental). Source the wrapper env first:

  ```sh
  eval $(scripts/hip_test.py --print-env)
  ```

  Authoritative runbook: [`docs/local_hip_environment.md`](../../../local_hip_environment.md).

- **Out of scope (pre-existing, environmental):** 4 `gpu_correctness` tests fail
  with hiprtc `unknown type int8_t` on this box. Everything here is scoped to
  int64 / int32 to avoid that path.

## Part 1 — #689 / #690 runtime confirmation

Five Surf fixtures (`*.ch`) plus their C++ driver mains (`main_*.cpp`). Each
driver allocates input tensors via the chelis runtime, calls the emitted entry,
and reads the output buffer with the **correct** element type (int64 via
`(int64_t*)t->data`, f32 via `t->data`), so the corruption is visible.

| fixture | driver | what it shows |
|---|---|---|
| `p689_neg_i64.ch` | `main_p689_neg.cpp` | #689: `neg` emits `kernel_neg_f32` over a `CHELIS_I64` buffer → garbage + dropped upper lanes |
| `p689_sum_i64.ch` | `main_p689_sum.cpp` | #689: `sum` emits `kernel_sum_ax0_f32` (f32 accumulator) over int64 → off by orders of magnitude |
| `p690_div_i64.ch` | `main_p690_div.cpp` | #690: `trunc_div` emits a correct `kernel_trunc_div_i64` but with **no divisor guard** → silent garbage on ÷0, no abort |
| `sanity_neg_f32.ch` | `main_sanity_neg_f32.cpp` | control: f32 `neg` is correct (harness is sound) |
| `control_add_i64.ch` | `main_control_add_i64.cpp` | control: int64 `add` (`kernel_add_i64`) is exact incl. above 2⁵³ (bug is op-specific, not a blanket int64 failure) |

### Reproduce one probe

```sh
eval $(scripts/hip_test.py --print-env)
cargo build -p chelis-cli
D=docs/investigations/probes/hip_runtime

target/debug/chelis build $D/p689_neg_i64.ch --target hip --output /tmp/p689
cp $D/main_p689_neg.cpp /tmp/p689/main.cpp
cd /tmp/p689 && hipcc -O2 -I. main.cpp p689_neg_i64_hip.cpp -L. \
    -lchelis_runtime -lhiprtc -lpthread -ldl -o bin && ./bin
```

Compare each against the evaluator, e.g.
`chelis eval 'neg(cast([16777217, 16777219, 9007199254740993, -5], int64))'`
(correct) vs `chelis eval 'trunc_div(cast([100,42,-7,9], int64), cast([0,0,0,3], int64))'`
(branded `error: integer division or remainder by zero`, exit 1 — the HIP binary
exits 0 with garbage).

## Part 2 — `hip_overflow_trap_spike.cpp` (#729 Phase-1 freeze, condition 1b)

Standalone single-file HIP spike measuring per-element integer-overflow
**detection** cost, mirroring Robert's Metal spike (#737) so the overhead
percentages compare. Atomic flag + first-failing-index detection; GPU-event
timing; median-of-15 after 3 warmup, checked/unchecked interleaved. Memory-bound
cells at 2²⁴ elements, compute-bound at 2²² × 64 chained ops with a runtime mask
so the backend cannot delete the checks. Self-checking: memcmp + in-range-clear
gate all 10 cells; the 6 memory-bound cells add planted-overflow-index and
host-oracle checks (prints `correctness: all cells passed`).

```sh
eval $(scripts/hip_test.py --print-env)
hipcc -O2 docs/investigations/probes/hip_runtime/hip_overflow_trap_spike.cpp \
    -o /tmp/hip_trap_spike && /tmp/hip_trap_spike
```

Result (this box): memory-bound detection is **free** (within ±2% timing noise —
bandwidth saturated); compute-bound ceilings run +100…+320%, applying only to hypothetical
fused kernels the HIP backend does not emit. Verdict: trap-everywhere is
feasible on HIP, matching the Metal conclusion. Full numbers on #736.
