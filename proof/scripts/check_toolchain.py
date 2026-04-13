#!/usr/bin/env python3
"""Verify the Chelis Proof local toolchain.

Checks each component documented in proof/toolchain.md and reports
the first missing piece. Exit 0 if everything is ready, non-zero on
any missing or misconfigured item.

Run:  python3 proof/scripts/check_toolchain.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py <3.11
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_ROOT = REPO_ROOT / "proof"
VIBE_CONFIG = Path.home() / ".vibe" / "config.toml"
VIBE_TRUSTED = Path.home() / ".vibe" / "trusted_folders.toml"

MIN_VIBE_VERSION = (2, 5, 0)
PINNED_LEAN_VERSION = "4.29.0"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_version(s: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 127, str(e)
    return r.returncode, (r.stdout + r.stderr).strip()


def _load_toml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return tomllib.load(f)


def check_elan() -> CheckResult:
    if not shutil.which("elan"):
        return CheckResult("elan", False, "not on PATH; install via elan-init.sh")
    return CheckResult("elan", True, "present")


def check_lean() -> CheckResult:
    if not shutil.which("lean"):
        return CheckResult("lean", False, "not on PATH")
    code, out = _run(["lean", "--version"])
    if code != 0:
        return CheckResult("lean", False, f"lean --version failed: {out}")
    if PINNED_LEAN_VERSION not in out:
        return CheckResult(
            "lean", False, f"expected {PINNED_LEAN_VERSION}, got: {out.splitlines()[0]}"
        )
    return CheckResult("lean", True, out.splitlines()[0])


def check_lake() -> CheckResult:
    if not shutil.which("lake"):
        return CheckResult("lake", False, "not on PATH")
    code, out = _run(["lake", "--version"])
    return CheckResult("lake", code == 0, out.splitlines()[0] if out else "")


def check_vibe() -> CheckResult:
    if not shutil.which("vibe"):
        return CheckResult("vibe", False, "not on PATH; uv tool install mistral-vibe")
    code, out = _run(["vibe", "--version"])
    if code != 0:
        return CheckResult("vibe", False, f"vibe --version failed: {out}")
    parsed = parse_version(out)
    if parsed is None:
        return CheckResult("vibe", False, f"unparseable version: {out}")
    if parsed < MIN_VIBE_VERSION:
        return CheckResult(
            "vibe",
            False,
            f"{out.strip()} < required {'.'.join(map(str, MIN_VIBE_VERSION))}",
        )
    return CheckResult("vibe", True, out.strip())


def check_lean_lsp_mcp() -> CheckResult:
    if not shutil.which("lean-lsp-mcp"):
        return CheckResult(
            "lean-lsp-mcp", False, "not on PATH; uv tool install lean-lsp-mcp"
        )
    code, out = _run(["lean-lsp-mcp", "--version"])
    return CheckResult("lean-lsp-mcp", code == 0, out.strip() or "present")


def check_api_key() -> CheckResult:
    if not os.environ.get("MISTRAL_API_KEY"):
        return CheckResult("MISTRAL_API_KEY", False, "env var not set")
    return CheckResult("MISTRAL_API_KEY", True, "set")


def check_vibe_config() -> CheckResult:
    data = _load_toml(VIBE_CONFIG)
    if data is None:
        return CheckResult("vibe config", False, f"{VIBE_CONFIG} missing")
    if not data.get("auto_approve"):
        return CheckResult("vibe config", False, "auto_approve must be true")
    if "lean" not in data.get("installed_agents", []):
        return CheckResult(
            "vibe config", False, '"lean" missing from installed_agents'
        )
    mcp = data.get("mcp_servers", [])
    if not any(
        s.get("name") == "lean-lsp" and s.get("transport") == "stdio" for s in mcp
    ):
        return CheckResult(
            "vibe config", False, "lean-lsp stdio MCP server not configured"
        )
    return CheckResult("vibe config", True, "ok")


def check_trusted_folders() -> CheckResult:
    data = _load_toml(VIBE_TRUSTED)
    if data is None:
        return CheckResult("trusted folders", False, f"{VIBE_TRUSTED} missing")
    trusted = set(data.get("trusted", []))
    if str(REPO_ROOT) not in trusted:
        return CheckResult(
            "trusted folders", False, f"{REPO_ROOT} not in trusted list"
        )
    return CheckResult("trusted folders", True, "ok")


def check_pinned_toolchain_file() -> CheckResult:
    tc = PROOF_ROOT / "lean" / "lean-toolchain"
    if not tc.exists():
        return CheckResult("lean-toolchain file", False, f"{tc} missing")
    content = tc.read_text().strip()
    if f"v{PINNED_LEAN_VERSION}" not in content:
        return CheckResult(
            "lean-toolchain file",
            False,
            f"expected v{PINNED_LEAN_VERSION}, got {content}",
        )
    return CheckResult("lean-toolchain file", True, content)


CHECKS = (
    check_elan,
    check_lean,
    check_lake,
    check_vibe,
    check_lean_lsp_mcp,
    check_api_key,
    check_vibe_config,
    check_trusted_folders,
    check_pinned_toolchain_file,
)


def run_all() -> list[CheckResult]:
    return [c() for c in CHECKS]


def main() -> int:
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        print(f"  [{mark}] {r.name:<{width}}  {r.detail}")
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed. See proof/toolchain.md.")
        return 1
    print("\nAll toolchain checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
