# Local macOS Environment

This repository is also developed from a macOS (darwin-arm64) workstation. The
default toolchain works without special setup, but there is one recurring local
failure mode worth a runbook: macOS first-exec assessment (`syspolicyd` /
Gatekeeper) can degrade and make **freshly linked binaries hang or stall on
their first exec**. Every `cargo build` produces fresh binaries, so this
poisons exactly the edit-build-test loop. Two variants have been observed
(chelis#356 tracks the investigation):

- **Variant A - hard wedge**: an error loop in `syspolicyd`; first execs hang
  indefinitely.
- **Variant B - silent slow assessment**: no error log lines at all, but each
  fresh binary waits minutes before being admitted. Volume-induced: it has
  been reproduced on a 28-minute-old boot, triggered by a mass first-exec
  burst (workspace `clippy --all-targets` + `cargo nextest run --workspace`,
  ~130 fresh test binaries). **A reboot is a reprieve, not a fix.**

## Quick reference (is first-exec wedged right now?)

Run the preflight probe — it compiles a trivial C program into a temp dir and
execs it with a short timeout:

```sh
.venv/bin/python scripts/preflight_exec_probe.py
# Inside an active Devenv shell:
chelis-exec-preflight
```

Expected when healthy:

```text
exec ok (42 ms)
```

If it instead exits 1 ("appears wedged", Variant A) or 3 ("admitting
binaries slowly", Variant B; first exec succeeded but took longer than
`--warn-ms`, default 2000 ms), local exec results are unreliable. Apply the
mitigations below and use CI as the oracle in the meantime.

A passing probe is a point-in-time result, not a session clearance:
degradation is volume-induced and can begin under a later build burst
(observed: probe at 377 ms immediately post-boot; minutes-long stalls within
the hour once a workspace build mass-launched fresh binaries).

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

Three checks distinguish this from a code or harness regression:

1. Inspect the `syspolicyd` log stream:

   ```sh
   log show --predicate 'process == "syspolicyd"' --last 5m
   ```

   Variant A shows up as an error loop with these exact signatures:

   ```text
   Unable to initialize qtn_proc: 3
   dispatch_mig_server returned 268435459
   ```

   Observed against `syspolicyd` PID 289 on darwin-arm64, 2026-06-10.
   **Variant B logs nothing** - a quiet log does not rule this out.

2. Sample a hung process:

   ```sh
   sample <pid>
   ```

   A stalled first-exec is parked in `_dyld_start` — it never reached the
   program's own code. If the sample shows frames inside the test binary
   instead, the hang is not this failure mode.

3. Check CPU accounting (Variant B's clearest fingerprint):

   ```sh
   ps -p <pid> -o etime,cputime   # minutes of etime, ~0:00.01 cputime
   ps -p $(pgrep -x syspolicyd) -o %cpu,cputime
   ```

   Stalled binaries accumulate essentially zero CPU while `syspolicyd` runs
   sustained 50-80% CPU. Measured 2026-06-11: an 84 MB Rust test binary took
   15-21 minutes to admit; small clang-compiled C binaries cleared in ~0.2 s
   (admission cost correlates with binary size); the assessment queue kept
   draining (syspolicyd busy) long after all exec load stopped.

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

3. **Avoid mass first-exec bursts.** Do not run full-workspace nextest
   locally during heavy agent sessions on this machine; the macOS Smoke CI
   job is the workspace oracle (next section). Single-binary work
   (`chelis` CLI on an already-assessed build, clippy, fmt, lint) is
   unaffected.
4. **Reboot.** Clears the current backlog, but is a reprieve, not a fix:
   Variant B has been reproduced within an hour of a fresh boot under
   first-exec volume.
5. **Durable fix (VERIFIED on this workstation 2026-06-11, chelis#356):
   Developer Tools exemption.** macOS exempts processes spawned by apps listed under
   System Settings > Privacy & Security > Developer Tools from the
   first-run malware scan. The app that needs the exemption is the
   RESPONSIBLE APP for the session's processes - NOT necessarily
   Terminal.app. Identify it by walking the process ancestry to the app
   directly under launchd:

   ```sh
   P=$$; while [ "$P" != "1" ]; do ps -o pid=,comm= -p $P; P=$(ps -o ppid= -p $P | tr -d ' '); done
   ```

   For sessions in a VS Code integrated terminal the chain ends at
   `Visual Studio Code`, so VS Code is the app to exempt (the
   `spctl developer-mode enable-terminal` CLI variant targets Terminal.app
   and does not help there). After toggling, FULLY quit and relaunch the
   responsible app (Cmd+Q; a new terminal tab is not enough). Before
   toggling, record whether the app was ALREADY exempted - if degradation
   occurred while exempted, the fix is refuted for that app. Verification
   protocol once enabled: re-check the ancestry, run the preflight probe,
   then a deliberate burst (`cargo nextest run -p chelis-compiler-api`
   after a `touch` rebuild) and compare admission behavior against the
   chelis#356 baselines; report on chelis#356.

   Verification result (2026-06-11, VS Code exempted + fully relaunched):
   a touch-rebuild of chelis-compiler-api followed by its suite admitted
   ~25 fresh test binaries and ran 264 tests in 1.05 s (vs 223 s wall for
   ONE binary the night before); the full-workspace burst (~130 fresh
   binaries, the canonical degradation trigger) ran 1,141+ tests with
   zero admission stalls while `syspolicyd` fell from 86% to ~8% CPU
   during the mass-exec phase. Heavyweight end-to-end tests were slow
   under CPU contention but always progressing (real accumulated CPU,
   never parked in `_dyld_start`).

   Security trade-off of the exemption (understand before enabling):
   processes spawned by the exempted apps skip the Gatekeeper/XProtect
   FIRST-LAUNCH malware assessment of unsigned/un-notarized executables.
   For locally compiled artifacts this loses almost nothing (the scan is
   signature-based; freshly built binaries match no signature). The real
   residual risk is DOWNLOADED prebuilt binaries run from the exempted
   shells (npm/pip postinstall payloads, `curl | sh` installers, binaries
   committed to repos): known malware that the first-exec scan would have
   flagged now runs immediately, with only slower background XProtect
   scans behind it. Background/behavioral XProtect, quarantine attributes,
   TCC prompts, and SIP all remain active; non-exempted apps are
   unaffected. Reasonable trade for a development workstation; not
   recommended on a general-purpose machine. The exemption is
   per-responsible-app and machine-local: nothing in this repo can
   enforce or verify it, so the volume-reduction practice (CI owns the
   workspace suite; see chelis#360) remains the primary discipline. The
   division of labor, post-verification: single binaries on a healthy
   queue always admitted in milliseconds, and the inner loop only ever
   suffered as collateral damage of mass bursts - so CI-first removes the
   common trigger, while the exemption removes the failure mode
   (degradation caused by anything else on the machine, or by a
   discipline slip, no longer stalls this workflow).

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
- exits 0 and prints `exec ok (N ms)` when the first exec completes
  promptly;
- exits 1 with a warning pointing at this runbook when the first exec times
  out (Variant A, the wedge classification);
- exits 3 with a warning when the first exec succeeds but takes longer than
  `--warn-ms` (default 2000 ms; Variant B, slow admission - healthy first
  execs are well under one second);
- exits 2 when the probe could not run at all (`cc` missing, the compile
  failed, the probe binary was not executable, or it exited non-zero) — an
  environment problem, not a degradation verdict;
- always cleans up its temp dir, so it is safe to run from anywhere.

Tests: `scripts/test_preflight_exec_probe.py`
(`.venv/bin/python scripts/test_preflight_exec_probe.py`).
