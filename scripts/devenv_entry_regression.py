#!/usr/bin/env python3
"""Reproduce concurrent Devenv entry in an isolated temporary checkout."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FAILURE_MARKERS = (
    "devenv:entershell failed",
    "dependency failed",
)


@dataclass(frozen=True)
class EntryResult:
    returncode: int
    output: str
    marker: str


def classify_entry(result: EntryResult) -> str | None:
    lowered = result.output.lower()
    payload_ran = result.marker in result.output
    if "load-exports" in lowered and (
        "no such file" in lowered or "cannot access" in lowered
    ):
        return "shared load-exports race returned"
    task_failure = any(marker in lowered for marker in FAILURE_MARKERS) or (
        re.search(r"running tasks in[^\n]*\(failed\)", lowered) is not None
    )
    if payload_ran and (result.returncode != 0 or task_failure):
        return "fail-open: payload ran after an entry-task failure"
    if result.returncode == 0 and not payload_ran:
        return "entry exited zero without payload"
    if payload_ran and result.returncode == 0:
        return None
    if not payload_ran and result.returncode != 0:
        return None
    return "inconsistent concurrent entry result"


def run_entry(devenv: str, checkout: Path, index: int) -> EntryResult:
    marker = f"CHELIS_DEVENV_PAYLOAD_{index}"
    completed = subprocess.run(
        [
            devenv,
            "shell",
            "--no-reload",
            "--",
            "python",
            "-c",
            f"print('{marker}')",
        ],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=600,
    )
    return EntryResult(completed.returncode, completed.stdout, marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.workers < 2:
        parser.error("--workers must be at least 2")
    devenv = shutil.which("devenv")
    if devenv is None:
        raise RuntimeError("devenv is not available")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="chelis-devenv-entry-") as raw_directory:
        checkout = Path(raw_directory) / "checkout"
        subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", "--no-checkout", str(REPO_ROOT), str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", head],
            cwd=checkout,
            check=True,
        )
        warm = run_entry(devenv, checkout, -1)
        warm_error = classify_entry(warm)
        if warm_error:
            raise RuntimeError(f"warm-up entry failed: {warm_error}\n{warm.output}")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=arguments.workers
        ) as pool:
            results = list(
                pool.map(
                    lambda index: run_entry(devenv, checkout, index),
                    range(arguments.workers),
                )
            )
        failures = [
            (index, reason, results[index].output)
            for index in range(arguments.workers)
            if (reason := classify_entry(results[index])) is not None
        ]
        if failures:
            details = "\n".join(
                f"entry {index}: {reason}\n{output}" for index, reason, output in failures
            )
            raise RuntimeError(f"concurrent Devenv entry regression failed:\n{details}")

    print(f"concurrent Devenv entry: PASS ({arguments.workers} workers at {head})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
