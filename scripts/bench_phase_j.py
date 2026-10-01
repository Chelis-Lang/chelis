#!/usr/bin/env python3
"""Phase J. Bench harness for the Compiled Artifact Caching plan.

Runs the headline `chelis test`/`chelis eval`/`chelis check` benchmarks
against the real Coral checkout (and chelis-std self-test corpus) using a
provided chelis binary. Captures wall-clock numbers and emits a markdown
table for the active performance summary or an archived baseline note.

Usage:
    python3 scripts/bench_phase_j.py \\
        --binary /path/to/target/release/chelis \\
        --label post \\
        --coral-dir /path/to/coral \\
        --chelis-repo /path/to/chelis \\
        --out /tmp/chelis-bench/post.json

Two runs are recommended: one against the post-cache build (this branch's
target/release/chelis) and one against the pre-cache baseline (binary built
from c13ea7a, the commit before the Wave 1 amortization landed).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchResult:
    name: str
    command: list[str]
    cwd: str
    wall_seconds: float
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    extra: dict = field(default_factory=dict)


def run_one(name: str, cmd: list[str], cwd: Path, timeout: float) -> BenchResult:
    print(f"[bench] {name}: {shlex.join(cmd)} (cwd={cwd})", flush=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        wall = time.monotonic() - t0
        exit_code = proc.returncode
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
    except subprocess.TimeoutExpired as exc:
        wall = time.monotonic() - t0
        exit_code = -1
        stdout_tail = (exc.stdout or "")[-2000:] if exc.stdout else ""
        stderr_tail = (exc.stderr or "")[-2000:] if exc.stderr else ""
        stderr_tail = (stderr_tail + f"\n[bench] TIMEOUT after {timeout}s")[:2000]
    print(f"[bench] {name}: wall={wall:.2f}s exit={exit_code}", flush=True)
    return BenchResult(
        name=name,
        command=cmd,
        cwd=str(cwd),
        wall_seconds=wall,
        exit_code=exit_code,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def parse_pass_fail(stdout_tail: str) -> Optional[tuple[int, int]]:
    # `chelis test` prints `<n> passed, <m> failed` as the final line for the suite.
    for line in reversed(stdout_tail.splitlines()):
        line = line.strip()
        if "passed" in line and "failed" in line:
            try:
                # e.g. "63 passed, 0 failed"
                parts = line.replace(",", "").split()
                p = int(parts[0])
                f = int(parts[2])
                return (p, f)
            except (ValueError, IndexError):
                continue
    return None


def run_benches(
    *,
    binary: Path,
    coral_dir: Path,
    chelis_repo: Path,
    nautilus_dir: Optional[Path],
    timeout: float,
    skip_coral_test: bool,
    skip_chelis_std: bool,
    label: str,
) -> dict:
    results: list[BenchResult] = []

    binary = binary.resolve()
    coral_dir = coral_dir.resolve()
    chelis_repo = chelis_repo.resolve()

    # 1. Coral chelis test tests/ — the headline
    if not skip_coral_test:
        r = run_one(
            "coral_test",
            [str(binary), "test", "tests/"],
            cwd=coral_dir,
            timeout=timeout,
        )
        pf = parse_pass_fail(r.stdout_tail)
        if pf:
            r.extra["passed"] = pf[0]
            r.extra["failed"] = pf[1]
        results.append(r)

    # 2. Coral chelis check src/core.ch
    r = run_one(
        "coral_check",
        [str(binary), "check", "src/core.ch"],
        cwd=coral_dir,
        timeout=timeout,
    )
    results.append(r)

    # 3. Coral chelis eval --file — pick a small test as a stand-in eval program
    eval_target = coral_dir / "tests" / "internal.ch"
    if eval_target.exists():
        r1 = run_one(
            "coral_eval_cold",
            [str(binary), "eval", "--file", str(eval_target)],
            cwd=coral_dir,
            timeout=timeout,
        )
        results.append(r1)
        # Warm = second invocation. No disk cache wired today; warm == second cold.
        r2 = run_one(
            "coral_eval_warm",
            [str(binary), "eval", "--file", str(eval_target)],
            cwd=coral_dir,
            timeout=timeout,
        )
        results.append(r2)

    # 4. chelis-std self-test corpus
    if not skip_chelis_std:
        std_tests = chelis_repo / "packages" / "chelis-std" / "tests"
        if std_tests.exists():
            r = run_one(
                "chelis_std_self_test",
                [str(binary), "test", str(std_tests)],
                cwd=chelis_repo / "packages" / "chelis-std",
                timeout=timeout,
            )
            pf = parse_pass_fail(r.stdout_tail)
            if pf:
                r.extra["passed"] = pf[0]
                r.extra["failed"] = pf[1]
            results.append(r)

    # 5. Nautilus optional
    if nautilus_dir and (nautilus_dir / "tests").exists():
        r = run_one(
            "nautilus_test",
            [str(binary), "test", "tests/"],
            cwd=nautilus_dir,
            timeout=timeout,
        )
        pf = parse_pass_fail(r.stdout_tail)
        if pf:
            r.extra["passed"] = pf[0]
            r.extra["failed"] = pf[1]
        results.append(r)

    return {
        "label": label,
        "binary": str(binary),
        "results": [asdict(r) for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--label", required=True, help="post | pre | other tag")
    ap.add_argument("--coral-dir", type=Path, required=True)
    ap.add_argument("--chelis-repo", type=Path, required=True)
    ap.add_argument("--nautilus-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="per-bench timeout in seconds (default 30 min)",
    )
    ap.add_argument(
        "--skip-coral-test",
        action="store_true",
        help="skip the Coral test suite (use when baseline would take too long)",
    )
    ap.add_argument(
        "--skip-chelis-std",
        action="store_true",
        help="skip the chelis-std self-test corpus",
    )
    args = ap.parse_args(argv)

    out = run_benches(
        binary=args.binary,
        coral_dir=args.coral_dir,
        chelis_repo=args.chelis_repo,
        nautilus_dir=args.nautilus_dir,
        timeout=args.timeout,
        skip_coral_test=args.skip_coral_test,
        skip_chelis_std=args.skip_chelis_std,
        label=args.label,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
