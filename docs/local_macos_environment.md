# Local macOS Environment

This repository is also developed from a macOS (darwin-arm64) workstation. The
default toolchain works without special setup, but there is one recurring local
failure mode worth a runbook: macOS first-exec assessment (`syspolicyd` /
Gatekeeper) can enter an error loop that makes **freshly linked binaries hang
on their first exec**. Every `cargo build` produces fresh binaries, so this
poisons exactly the edit-build-test loop.

## Quick reference (is first-exec wedged right now?)

Run the preflight probe — it compiles a trivial C program into a temp dir and
execs it with a short timeout:

```sh
.venv/bin/python scripts/preflight_exec_probe.py
```

Expected when healthy:

```text
exec ok (42 ms)
```

If it instead exits 1 with a "first-exec assessment appears wedged" warning,
local exec results are unreliable. Apply the mitigations below and use CI as
the oracle in the meantime.

The sections below explain the symptom, the diagnosis, and the mitigations;
read them when the loop looks wedged, not on the happy path.

## Symptom

- Freshly linked binaries hang in `_dyld_start` on their **first** exec, for
  minutes or indefinitely. The process starts but never reaches `main`.
- The effect is worst under parallel first-exec bursts — e.g. `cargo nextest`
  listing ~30 newly built test binaries at once. Suites that could not
  complete locally when this struck: the workspace-level
  `cargo nextest run -p chelis-compiler-api` and the full `-p chelis-cli`
  runs.
- The edit-build-test loop appears wedged: every rebuild produces fresh
  binaries, and each fresh binary pays the hang again.
- Already-assessed binaries run fine. Re-running a binary that has executed
  once before is unaffected, which is why the problem can masquerade as a
  flaky build or a hung test rather than an OS-level issue.

## Diagnosis

Two checks distinguish this from a code or harness regression:

1. Inspect the `syspolicyd` log stream:

   ```sh
   log show --predicate 'process == "syspolicyd"' --last 5m
   ```

   The wedge shows up as an error loop with these exact signatures:

   ```text
   Unable to initialize qtn_proc: 3
   dispatch_mig_server returned 268435459
   ```

   Observed against `syspolicyd` PID 289 on darwin-arm64, 2026-06-10.

2. Sample a hung process:

   ```sh
   sample <pid>
   ```

   A wedged first-exec is parked in `_dyld_start` — it never reached the
   program's own code. If the sample shows frames inside the test binary
   instead, the hang is not this failure mode.

## Mitigations

In order of preference:

1. **Wait.** The loop has been observed to self-clear; in the one measured
   incident it cleared after ~25 minutes, then returned under load. If the
   probe starts passing again, the window is open.
2. **Serialize first-execs.** The wedge is load-sensitive. Run suites one at a
   time and pass `--test-threads=1` so only one fresh binary is being assessed
   at any moment:

   ```sh
   cargo nextest run -p <crate> -- --test-threads=1
   ```

3. **Reboot.** Assumed reliable: it restarts `syspolicyd` and clears the error
   loop. Use this when the wait-or-serialize options are not viable.

## CI Is the Fallback Oracle

When local exec is wedged, do not block on the local run: the
`macos-smoke` CI job (`.github/workflows/ci.yml`) runs the full workspace
test suite on macOS and serves as the macOS signal. Push the branch and
let CI serve as the oracle, noting in the PR or phase docs that local
validation was blocked by this failure mode.

## Preflight Probe Details

`scripts/preflight_exec_probe.py` is the documented manual command for this
runbook. It:

- writes a trivial C file into a private temp dir, compiles it with `cc`
  (the compile stage carries its own 120s timeout, so a hung toolchain is
  also bounded), and execs the result with a configurable timeout
  (`--timeout`, default 15 seconds);
- exits 0 and prints `exec ok (N ms)` when the first exec completes;
- exits 1 with a warning pointing at this runbook when the first exec times
  out (the wedge classification);
- exits 2 when the probe could not run at all (`cc` missing, the compile
  failed, the probe binary was not executable, or it exited non-zero) — an
  environment problem, not a wedge verdict;
- always cleans up its temp dir, so it is safe to run from anywhere.

Tests: `scripts/test_preflight_exec_probe.py`
(`.venv/bin/python scripts/test_preflight_exec_probe.py`).
