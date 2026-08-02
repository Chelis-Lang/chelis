"""Acceptance probe for chelis#914: bindings SIGINT -> KeyboardInterrupt latency.

Before the fix, `chelis.eval` ran the whole evaluation inside
`py.allow_threads` on the Python main thread. SIGINT set Python's handler flag
promptly, but `KeyboardInterrupt` cannot be raised until control returns to the
interpreter's eval loop -- which, on that code shape, was after the evaluation
had already finished. Interrupt latency therefore equalled the full run time:
for the workloads in chelis#828, minutes.

After the fix, the evaluation runs on a worker thread with a cancellation token
installed while the calling thread polls `py.check_signals()` every 50 ms.

**Acceptance: SIGINT to KeyboardInterrupt in under 250 ms.**

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

# How long to let the evaluation run before interrupting it. Long enough to be
# unambiguously past compilation and inside the interpreter.
SIGINT_AFTER_S = 5.0

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


def measure_interrupt_latency() -> tuple[float | None, str]:
    """SIGINT a running `chelis.eval`; return (latency_ms, child stderr)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(source=SLOW_PROGRAM)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Own process group: signal only the child, and do not inherit this
        # process's signal disposition.
        start_new_session=True,
    )

    # Wait until the child has imported chelis and entered the evaluation, so
    # we are timing an interrupt of running work rather than of startup.
    while True:
        line = proc.stderr.readline()
        if not line or line.strip() == "READY":
            break
    time.sleep(SIGINT_AFTER_S)

    try:
        sent = time.monotonic()
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        _, err = proc.communicate()
        return None, f"child exited before it could be interrupted: {err.strip()}"

    try:
        _, err = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None, "child never returned after SIGINT"

    return (time.monotonic() - sent) * 1000.0, err.strip()


def check_interrupt_latency() -> bool:
    latency_ms, detail = measure_interrupt_latency()
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
        check_uninterrupted_eval_is_unaffected,
        check_evaluation_error_still_surfaces,
    ]
    results = [check() for check in checks]
    passed = all(results)
    print(f"\n{'ALL PASS' if passed else 'FAILURES'}: {sum(results)}/{len(results)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
