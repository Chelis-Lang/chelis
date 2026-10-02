#!/usr/bin/env python3
"""Check that two runs of the bundle's build script pack the same chelis-std runtime.

`crates/chelis-std-bundle/build.rs` packs `packages/chelis-std` into the
crate's `OUT_DIR` while the compiler builds. This check exports the committed
`HEAD` tree with `git archive`, so no uncommitted or untracked file can reach
it, then runs `cargo check` on the crate there twice in one fresh target
directory. The first run compiles the packing code and runs the build script.
The second sets a codegen option on the bundle crate alone, which changes
nothing the build script computes but gives it a new `OUT_DIR`, so cargo
compiles and runs it again while reusing every dependency. The check requires
the two output directories to differ and the packed archive, shell, and version
to be byte-identical. The runs differ in their `OUT_DIR`, so in every path the
packer stages and writes through, and in their processes; a difference means
the embedded runtime depends on something other than the committed std sources
and the packing code.

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
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "chelis-std-bundle"
TARGET_ROOT = Path("target") / "std-bundle-reproducibility"
OUTPUTS = ("chelis-std.tar.zst", "chelis-std.chb", "chelis-std.version")
# Changes only the bundle crate's units, and none of their behavior, so the
# second run rebuilds and reruns the build script alone in a new OUT_DIR.
SECOND_RUN_CONFIG = f"profile.dev.package.{PACKAGE}.codegen-units=1"


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


def export_head(repo: Path, destination: Path) -> Path:
    """Write the tree `HEAD` commits in `repo` to a fresh `destination`."""
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(destination, filter="data")
    return destination


def build(source: Path, target_dir: Path, *extra: str) -> tuple[Path, dict[str, bytes]]:
    """Check the bundle crate of the workspace at `source` in `target_dir`, and
    return its build script's `OUT_DIR` and outputs."""
    argv = [
        "cargo",
        "check",
        "--locked",
        "-p",
        PACKAGE,
        "--target-dir",
        str(target_dir),
        *extra,
        "--message-format",
        "json-render-diagnostics",
    ]
    completed = subprocess.run(argv, cwd=source, stdout=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise ReproducibilityError(f"{shlex.join(argv)} exited {completed.returncode}")
    out_dir = bundle_out_dir(completed.stdout)
    return out_dir, {name: (out_dir / name).read_bytes() for name in OUTPUTS}


def build_twice(source: Path, target_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Run the build script in a fresh `target_dir`, then again in a new
    `OUT_DIR` there, and return both runs' outputs."""
    shutil.rmtree(target_dir, ignore_errors=True)
    first_dir, first = build(source, target_dir)
    second_dir, second = build(source, target_dir, "--config", SECOND_RUN_CONFIG)
    if second_dir == first_dir:
        raise ReproducibilityError(
            f"the second run reported the first run's OUT_DIR {first_dir}, so "
            f"`--config {SECOND_RUN_CONFIG}` did not make cargo run the build "
            "script again"
        )
    return first, second


def differences(first: dict[str, bytes], second: dict[str, bytes]) -> list[str]:
    """One line per output whose bytes differ between the two runs."""
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
    root = REPO_ROOT / TARGET_ROOT
    try:
        source = export_head(REPO_ROOT, root / "source")
        builds = build_twice(source, root / "target")
    except (ReproducibilityError, OSError, subprocess.CalledProcessError) as error:
        print(f"std bundle reproducible: error: {error}", file=sys.stderr)
        print("std bundle reproducible: FAIL")
        return 1
    found = differences(*builds)
    if found:
        print(
            "std bundle reproducible: two runs packed different chelis-std bytes:",
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
