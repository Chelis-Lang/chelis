#!/usr/bin/env python3
"""Check that two independent builds embed the same chelis-std runtime.

`crates/chelis-std-bundle/build.rs` packs `packages/chelis-std` into the
crate's `OUT_DIR` while the compiler builds. This check runs `cargo check` on
the crate, which compiles and runs the build script, in two fresh target
directories and requires the packed archive, shell, and version to be
byte-identical. The builds differ in their target directory, so
in every path the packer stages and writes through, and in their processes; a
difference means the embedded runtime depends on something other than the std
sources and the packing code.

Agreement with `chelis reef build` is a separate property, owned by the
`building_the_std_sources_reproduces_the_embedded_runtime` test in
`crates/chelis-cli/tests/bundled_chelis_std_loader.rs`.

Usage:

    <managed-python> scripts/check_std_bundle_reproducible.py

Acceptance is exit 0 with the final line ``std bundle reproducible: PASS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "chelis-std-bundle"
TARGET_ROOT = Path("target") / "std-bundle-reproducibility"
OUTPUTS = ("chelis-std.tar.zst", "chelis-std.chb", "chelis-std.version")


class ReproducibilityError(RuntimeError):
    """A build failed or did not report the bundle's build-script output."""


def bundle_out_dir(messages: str) -> Path:
    """The `OUT_DIR` cargo reports for the bundle's build script."""
    found = []
    for line in messages.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "build-script-executed":
            continue
        package_id = message.get("package_id", "")
        if (
            f"/{PACKAGE}#" in package_id
            or f"#{PACKAGE}@" in package_id
            or package_id.startswith(f"{PACKAGE} ")
        ):
            found.append(Path(message["out_dir"]))
    if len(found) != 1:
        raise ReproducibilityError(
            f"expected one build-script output for {PACKAGE}, found {len(found)}"
        )
    return found[0]


def build(target_dir: Path) -> dict[str, bytes]:
    """Run the bundle's build script from scratch in `target_dir` and read its
    outputs."""
    shutil.rmtree(target_dir, ignore_errors=True)
    completed = subprocess.run(
        [
            "cargo",
            "check",
            "--locked",
            "-p",
            PACKAGE,
            "--target-dir",
            str(target_dir),
            "--message-format",
            "json-render-diagnostics",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ReproducibilityError(
            f"cargo check -p {PACKAGE} --target-dir {target_dir} "
            f"exited {completed.returncode}"
        )
    out_dir = bundle_out_dir(completed.stdout)
    return {name: (out_dir / name).read_bytes() for name in OUTPUTS}


def differences(first: dict[str, bytes], second: dict[str, bytes]) -> list[str]:
    """One line per output whose bytes differ between the two builds."""
    lines = []
    for name in OUTPUTS:
        if first[name] != second[name]:
            lines.append(
                f"{name}: {hashlib.sha256(first[name]).hexdigest()} != "
                f"{hashlib.sha256(second[name]).hexdigest()}"
            )
    return lines


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    targets = [REPO_ROOT / TARGET_ROOT / name for name in ("first", "second")]
    try:
        builds = [build(target) for target in targets]
    except (ReproducibilityError, OSError) as error:
        print(f"std bundle reproducible: error: {error}", file=sys.stderr)
        print("std bundle reproducible: FAIL")
        return 1
    found = differences(*builds)
    if found:
        print(
            "std bundle reproducible: two builds embed different chelis-std bytes:",
            file=sys.stderr,
        )
        for line in found:
            print(f"  {line}", file=sys.stderr)
        print("std bundle reproducible: FAIL")
        return 1
    for name in OUTPUTS:
        print(f"{name}: {hashlib.sha256(builds[0][name]).hexdigest()}")
    print("std bundle reproducible: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
