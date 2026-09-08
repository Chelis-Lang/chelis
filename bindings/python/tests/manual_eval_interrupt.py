"""Acceptance probe for chelis#914 + chelis#930: bindings SIGINT latency.

Before chelis#914, `chelis.eval` ran the whole job inside `py.allow_threads` on
the Python main thread. SIGINT set Python's handler flag promptly, but
`KeyboardInterrupt` cannot be raised until control returns to the interpreter's
eval loop -- which, on that code shape, was after the job had already finished.
Interrupt latency therefore equalled the full run time: for the workloads in
chelis#828, minutes.

After that fix, the job runs on a worker thread with a cancellation token
installed while the calling thread polls `py.check_signals()` every 50 ms.

chelis#914 covered the EVALUATION phase only, because cancellation was observed
at node visits and the front end has none. A SIGINT arriving during parse /
desugar / check / lower still waited out the remaining compile: measured on a
70 KB source with a ~19.3 s front end, an interrupt at t+2 s raised at t+19.32 s.
chelis#930 added token checks at the front end's phase boundaries and, inside
the passes that scale with declaration count, per top-level declaration.

**Acceptance:**

* evaluation phase (chelis#914) -- SIGINT to `KeyboardInterrupt` under 250 ms;
* front-end phase (chelis#930) -- under 1 s, and a small fraction of the
  compile that was abandoned. The second half matters more than the absolute
  number: latency tracking "time left in the compile" is the pre-fix signature.

Run (see `crates/chelis-python/tests/manual_eval_interrupt.rs` for the wrapper
that installs the bindings first):

    .venv/bin/python bindings/python/tests/manual_eval_interrupt.py

Exits 0 when every check passes, 1 otherwise, printing a PASS/FAIL line each.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

BUDGET_MS = 250.0

# chelis#930's own stated acceptance for the front-end case. Looser than the
# evaluation budget because the residual uninterruptible unit is one top-level
# declaration rather than one node visit.
FRONTEND_BUDGET_MS = 1000.0

# How long to let the evaluation run before interrupting it. Long enough to be
# unambiguously past compilation and inside the interpreter.
SIGINT_AFTER_S = 5.0

# Declaration count for the front-end-dominated program, matching chelis#930's
# repro (~70 KB of source, ~19 s front end in a debug build).
FRONTEND_DEFS = 1500

# Fraction of the measured uncancelled compile at which to interrupt. Deriving
# it from a measured baseline rather than hard-coding seconds keeps the probe
# honest across build profiles and machines: on a fast build a fixed 2 s would
# land after the compile finished and the probe would silently measure nothing.
FRONTEND_SIGINT_AT_FRACTION = 0.25

# A nested fold: ~4M interpreter iterations, so the uninterrupted run takes
# minutes, but no intermediate list exceeds 2000 elements.
#
# The nesting is load-bearing. Cancellation latency is bounded by the longest
# single uninterruptible step, and allocating or dropping one very large list
# is such a step -- a flat `range(0, 40_000_000)` measures list teardown, not
# the cancellation mechanism. Nesting keeps every allocation small, which is
# also the shape of a real iterative numerical kernel.
SLOW_PROGRAM = """def inner(seed: i64) -> i64 =
  fold(fn (acc, x) -> acc + x * seed, 0i64, range(0, 2000))

result = fold(fn (acc, k) -> acc + inner(k), 0i64, range(0, 2000))
"""

FAST_PROGRAM = "result = fold(fn (acc, x) -> acc + x, 0i64, range(0, 100))\n"


def front_end_heavy_program(defs: int = FRONTEND_DEFS) -> str:
    """chelis#930's repro shape: a long front end and a trivial evaluation.

    The evaluation is one multiply-add, so an interrupt observed only at node
    visits (chelis#914's guarantee) would have nothing to catch -- the compile
    would finish first and latency would equal the remaining compile time.
    Anything this probe measures is therefore front-end cancellation.
    """
    body = "\n".join(f"def f{i}(x: i64) -> i64 = x * {i}i64 + 1i64" for i in range(defs))
    return f"{body}\nresult = f0(1i64)\n"

# Runs in a child process: this probe needs a real SIGINT delivered to a
# process whose main thread is inside `chelis.eval`.
_CHILD = r"""
import sys
import chelis

SOURCE = {source!r}

sys.stderr.write("READY\n")
sys.stderr.flush()
try:
    chelis.eval(SOURCE)
except KeyboardInterrupt:
    sys.stderr.write("INTERRUPTED\n")
    sys.stderr.flush()
    sys.exit(130)
except BaseException as exc:  # noqa: BLE001 - diagnostic probe
    sys.stderr.write("OTHER {{}}: {{}}\n".format(type(exc).__name__, exc))
    sys.stderr.flush()
    sys.exit(2)
sys.stderr.write("COMPLETED\n")
sys.stderr.flush()
sys.exit(0)
"""


def _spawn(source: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(source=source)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Own process group: signal only the child, and do not inherit this
        # process's signal disposition.
        start_new_session=True,
    )


def _wait_for_ready(proc: subprocess.Popen[str]) -> None:
    """Block until the child has imported chelis and entered `chelis.eval`.

    Without this the probe would time an interrupt of interpreter startup
    rather than of running work.
    """
    while True:
        line = proc.stderr.readline()  # type: ignore[union-attr]
        if not line or line.strip() == "READY":
            return


def measure_uncancelled_runtime(source: str) -> float | None:
    """Seconds for `chelis.eval(source)` to finish undisturbed, or None."""
    proc = _spawn(source)
    _wait_for_ready(proc)
    started = time.monotonic()
    try:
        _, err = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    if "COMPLETED" not in err:
        return None
    return time.monotonic() - started


def measure_interrupt_latency(
    source: str, sigint_after_s: float
) -> tuple[float | None, str]:
    """SIGINT a running `chelis.eval`; return (latency_ms, child stderr)."""
    proc = _spawn(source)
    _wait_for_ready(proc)
    time.sleep(sigint_after_s)

    try:
        sent = time.monotonic()
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        _, err = proc.communicate()
        return None, f"child exited before it could be interrupted: {err.strip()}"

    try:
        _, err = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None, "child never returned after SIGINT"

    return (time.monotonic() - sent) * 1000.0, err.strip()


def check_interrupt_latency() -> bool:
    latency_ms, detail = measure_interrupt_latency(SLOW_PROGRAM, SIGINT_AFTER_S)
    if latency_ms is None:
        print(f"FAIL  sigint_raises_keyboardinterrupt  ({detail})")
        return False
    if "INTERRUPTED" not in detail:
        print(f"FAIL  sigint_raises_keyboardinterrupt  (child said {detail!r})")
        return False
    ok = latency_ms < BUDGET_MS
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict}  sigint_raises_keyboardinterrupt  "
        f"({latency_ms:.1f} ms, budget {BUDGET_MS:.0f} ms)"
    )
    return ok


def check_frontend_interrupt_latency() -> bool:
    """chelis#930: SIGINT during compilation, not during evaluation.

    Measures the undisturbed compile first, then interrupts a second run a
    quarter of the way in. Two assertions, because either alone is weak: an
    absolute budget on a fast machine can pass for the wrong reason, and a
    ratio alone would accept a slow absolute latency on a slow machine.
    """
    source = front_end_heavy_program()
    baseline_s = measure_uncancelled_runtime(source)
    if baseline_s is None:
        print("FAIL  sigint_during_front_end  (baseline run did not complete)")
        return False

    sigint_after_s = baseline_s * FRONTEND_SIGINT_AT_FRACTION
    latency_ms, detail = measure_interrupt_latency(source, sigint_after_s)
    if latency_ms is None:
        print(f"FAIL  sigint_during_front_end  ({detail})")
        return False
    if "INTERRUPTED" not in detail:
        print(f"FAIL  sigint_during_front_end  (child said {detail!r})")
        return False

    remaining_ms = (baseline_s - sigint_after_s) * 1000.0
    within_budget = latency_ms < FRONTEND_BUDGET_MS
    # The pre-fix behaviour was latency == remaining compile time. Requiring a
    # 10x margin is what distinguishes "cancelled" from "happened to be nearly
    # done".
    abandoned_the_rest = latency_ms * 10 < remaining_ms
    ok = within_budget and abandoned_the_rest
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict}  sigint_during_front_end  "
        f"({latency_ms:.1f} ms, budget {FRONTEND_BUDGET_MS:.0f} ms; "
        f"interrupted {sigint_after_s:.1f} s into a {baseline_s:.1f} s run, "
        f"{remaining_ms / 1000.0:.1f} s of compile abandoned)"
    )
    return ok


def check_uninterrupted_eval_is_unaffected() -> bool:
    """Negative parity: the worker-thread path must not change normal results."""
    import chelis

    result = chelis.eval(FAST_PROGRAM)
    values = [root.value for root in result.roots]
    ok = values == [4950]
    print(
        f"{'PASS' if ok else 'FAIL'}  uninterrupted_eval_returns_its_value  "
        f"(got {values})"
    )
    return ok


def check_evaluation_error_still_surfaces() -> bool:
    """Negative parity: a genuine error must not be reported as cancellation."""
    import chelis

    try:
        chelis.eval("result = no_such_function(1)\n")
    except Exception as exc:  # noqa: BLE001 - we assert on the message
        message = str(exc)
        ok = "cancel" not in message.lower()
        print(
            f"{'PASS' if ok else 'FAIL'}  real_error_is_not_reported_as_cancellation  "
            f"({message[:80]!r})"
        )
        return ok
    print("FAIL  real_error_is_not_reported_as_cancellation  (no error raised)")
    return False


def main() -> int:
    checks = [
        check_interrupt_latency,
        check_frontend_interrupt_latency,
        check_uninterrupted_eval_is_unaffected,
        check_evaluation_error_still_surfaces,
    ]
    results = [check() for check in checks]
    passed = all(results)
    print(f"\n{'ALL PASS' if passed else 'FAILURES'}: {sum(results)}/{len(results)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
