#!/usr/bin/env python3
"""Verify Chelis's executable axis kernel with pinned Verus.

Also require a deliberately false postcondition on a temporary copy of the
same source to fail for the expected proof reason.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile


RELEASE = "0.2026.09.27.3cf1832"
ASSETS = {
    ("Darwin", "arm64"): (
        "arm64-macos",
        "14ae529ef84178dd6b6debac76dd093803fcfd01123023561e19ac1575e53b83",
    ),
    ("Darwin", "x86_64"): (
        "x86-macos",
        "c304fd8fe2eb143cc5a615461fb37b17f00d2387d6054ec013717eb056c0b0ff",
    ),
    ("Linux", "x86_64"): (
        "x86-linux",
        "43814031e10df043221d83c7fa842e3ebd662cb7b0090fc1eb4d7a8673226978",
    ),
}
REPO = Path(__file__).resolve().parents[3]


def run(*args: str | Path, cwd: Path = REPO, env: dict[str, str] | None = None) -> None:
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def verifier_home() -> Path:
    try:
        asset, expected = ASSETS[(platform.system(), platform.machine())]
    except KeyError as error:
        raise SystemExit(f"unsupported Verus release platform: {error}") from error
    override = os.environ.get("CHELIS_VERUS_HOME")
    if override:
        home = Path(override)
        if not (home / "cargo-verus").is_file():
            raise SystemExit(f"CHELIS_VERUS_HOME lacks cargo-verus: {home}")
        return home

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "chelis/verus" / RELEASE
    home = cache / f"verus-{asset}"
    if (home / "cargo-verus").is_file():
        return home
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"verus-{asset}.zip"
    url = (
        "https://github.com/verus-lang/verus/releases/download/"
        f"release/{RELEASE}/verus-{RELEASE}-{asset}.zip"
    )
    run("curl", "-fL", "--retry", "3", "-o", archive, url)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Verus archive SHA-256 mismatch: {actual}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(cache)
    if not (home / "cargo-verus").is_file():
        raise SystemExit(f"Verus archive lacks cargo-verus: {home}")
    return home


def main() -> None:
    home = verifier_home()
    env = os.environ.copy()
    env["RUSTUP_NO_SELF_UPDATE"] = "1"
    env["RUSTUP_TOOLCHAIN"] = "1.98.1"
    env["PATH"] = f"{home}{os.pathsep}{env['PATH']}"
    toolchains = subprocess.run(
        ["rustup", "toolchain", "list"], capture_output=True, text=True, check=True, env=env
    ).stdout
    if not any(line.startswith("1.98.1-") for line in toolchains.splitlines()):
        run("rustup", "toolchain", "install", "1.98.1", "--profile", "minimal", env=env)
    run("cargo", "verus", "verify", "-p", "chelis-axis-core", env=env)

    with tempfile.TemporaryDirectory(prefix="chelis-axis-negative-") as temp:
        temp_root = Path(temp)
        (temp_root / "src").mkdir()
        shutil.copyfile(REPO / "crates/chelis-axis-core/src/lib.rs", temp_root / "src/lib.rs")
        verified = (REPO / "crates/chelis-axis-core/src/verified.rs").read_text()
        original = "valid == valid_permutation(axes@, rank)"
        if verified.count(original) != 1:
            raise SystemExit("cannot locate the exact permutation postcondition")
        (temp_root / "src/verified.rs").write_text(verified.replace(original, "valid == false"))
        (temp_root / "Cargo.toml").write_text(
            """[package]
name = "chelis-axis-negative-control"
version = "0.0.0"
edition = "2024"

[dependencies]
vstd = "=0.0.0-2026-09-20-0158"

[package.metadata.verus]
verify = true

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(verus_only)', 'cfg(verus_keep_ghost)'] }
"""
        )
        control = subprocess.run(
            [
                "cargo", "verus", "verify", "--manifest-path", str(temp_root / "Cargo.toml"),
                "--target-dir", str(REPO / "target"),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        if control.returncode == 0 or "postcondition not satisfied" not in control.stderr:
            print(control.stdout, control.stderr, file=sys.stderr)
            raise SystemExit("false postcondition was not rejected for the expected proof reason")
    print("Verus accepted the axis contracts and rejected the false postcondition")


if __name__ == "__main__":
    main()
