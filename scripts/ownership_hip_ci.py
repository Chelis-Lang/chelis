"""Run the unchanged Phase 3 hardware oracle and retain a commit-bound CI receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PASS_MARKER = "COMPILED VALUE OWNERSHIP PHASE 3: PASS"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def verify_head(root: Path, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("expected head must be a full lowercase 40-character commit SHA")
    actual = git(root, "rev-parse", "HEAD")
    if actual != expected:
        raise ValueError(f"checkout {actual} does not match requested commit {expected}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked checkout is dirty; an oracle receipt requires a clean commit")
    return actual


def run_logged(command: list[str], root: Path, log_path: Path) -> tuple[int, int]:
    last_line = ""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if line.strip():
                    last_line = line.strip()
            code = process.wait()
    return (code if code else int(last_line != PASS_MARKER), code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", default=os.environ.get("OWNERSHIP_HIP_COMMIT"))
    args = parser.parse_args(argv)
    if args.expected_head is None:
        parser.error("--expected-head or OWNERSHIP_HIP_COMMIT is required")
    output = ROOT / "target/ownership-hip-ci"
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/compiled_value_ownership_oracle.py",
               "--phase", "3", "--require-hip"]
    receipt = {"expected_head": args.expected_head, "command": command,
               "started_at": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "hipcc": shutil.which("hipcc"),
               "rocminfo": shutil.which("rocminfo"), "passed": False}
    result = 1
    try:
        receipt["head"] = verify_head(ROOT, args.expected_head)
        result, code = run_logged(command, ROOT, output / "oracle.log")
        receipt["oracle_exit_code"] = code
        # Reject a receipt if the oracle or a concurrent writer changed the source.
        verify_head(ROOT, args.expected_head)
        receipt["passed"] = result == 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result = 1
        receipt["error"] = str(error)
        print(f"OWNERSHIP HIP CI: FAIL: {error}", file=sys.stderr)
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
