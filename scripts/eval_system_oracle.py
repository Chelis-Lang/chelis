#!/usr/bin/env python3
"""Evaluator-only system boundary acceptance oracle.

From the repository root, run ``python3 scripts/eval_system_oracle.py``.
The script resolves the checkout from its own path, so an absolute script path
also works from other directories. Each leg executes against that checkout;
compiled C and the runtime C ABI retain their own acceptance suites.
"""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEGS = (
    (sys.executable, "scripts/test_eval_system_guard.py"),
    (sys.executable, "scripts/eval_system_guard.py"),
    (
        "cargo", "nextest", "run", "-p", "chelis-compiler-api", "--lib",
        "-E", "test(runtime::system_tests::) | test(runtime::eval::list_dir_conversion_tests::)",
        "--no-tests", "fail",
    ),
    ("cargo", "nextest", "run", "-p", "chelis-compiler-api", "--test", "invariant_decode", "--no-tests", "fail"),
    ("cargo", "nextest", "run", "-p", "chelis-compiler-api", "--test", "issue_1479_list_dir_order", "--no-tests", "fail"),
    ("cargo", "nextest", "run", "-p", "chelis-compiler-api", "--test", "eval_system_boundary", "--no-tests", "fail"),
    ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "issue_1479_list_dir_lane_parity", "--no-tests", "fail"),
    ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "process_run_builtin", "--no-tests", "fail"),
)


def main() -> int:
    for number, argv in enumerate(LEGS, 1):
        print(f"[{number}/{len(LEGS)}] {shlex.join(argv)}", flush=True)
        result = subprocess.run(argv, cwd=ROOT, check=False)
        if result.returncode:
            print(f"EVAL SYSTEM BOUNDARY ORACLE: FAIL (leg {number}, exit {result.returncode})", file=sys.stderr)
            return result.returncode
    print("EVAL SYSTEM BOUNDARY ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
