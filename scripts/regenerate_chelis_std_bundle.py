#!/usr/bin/env python3
"""Regenerate the chelis-std bundle artifacts that ship inside the
chelis binary.

This script is the canonical pipeline for refreshing the bytes that
`crates/chelis-std-bundle` embeds via `include_bytes!()`. Run it from
the repo root after any change to `packages/chelis-std/src/**.ch`,
then commit the updated files in `crates/chelis-std-bundle/dist/`.

Pipeline:
  1. cargo build -p chelis --release
  2. (release-binary) chelis reef build packages/chelis-std/
     produces packages/chelis-std/dist/chelis-std-<version>.{tar.zst,chb}
  3. copy those into crates/chelis-std-bundle/dist/
  4. git diff --stat to show what changed

Invocation:
  python3 scripts/regenerate_chelis_std_bundle.py [--debug]

Flags:
  --debug   build chelis in dev profile instead of release (faster
            iteration when wiring the script itself).

Exit codes:
  0  success (artifacts up-to-date or successfully regenerated)
  1  build/copy failed; stderr carries the underlying tool output.

The script is intentionally idempotent: re-running with no source
changes produces no diff. The committed artifacts ARE the source of
truth — the build.rs in chelis-std-bundle does not regenerate them.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def repo_root() -> Path:
    """The repository root, located by walking up from this script."""
    here = Path(__file__).resolve()
    # scripts/regenerate_chelis_std_bundle.py lives one level under repo root.
    return here.parent.parent


def chelis_std_version(repo: Path) -> str:
    """Read the chelis-std package version from packages/chelis-std/reef.toml."""
    manifest_path = repo / "packages" / "chelis-std" / "reef.toml"
    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)
    return manifest["package"]["version"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="build chelis in dev profile instead of release",
    )
    args = parser.parse_args()

    repo = repo_root()
    version = chelis_std_version(repo)
    bundle_dist = repo / "crates" / "chelis-std-bundle" / "dist"
    pkg_dist = repo / "packages" / "chelis-std" / "dist"
    archive_name = f"chelis-std-{version}.tar.zst"
    shell_name = f"chelis-std-{version}.chb"

    bundle_dist.mkdir(parents=True, exist_ok=True)

    profile_flag = [] if args.debug else ["--release"]
    profile_dir = "debug" if args.debug else "release"

    # Step 1: build the chelis CLI.
    print(f"[1/4] cargo build -p chelis-cli {' '.join(profile_flag)}", file=sys.stderr)
    rc = subprocess.run(
        ["cargo", "build", "-p", "chelis-cli", *profile_flag],
        cwd=repo,
    ).returncode
    if rc != 0:
        print("ERROR: cargo build failed", file=sys.stderr)
        return 1

    chelis_bin = repo / "target" / profile_dir / "chelis"
    if not chelis_bin.is_file():
        print(
            f"ERROR: chelis binary not found at {chelis_bin} after build",
            file=sys.stderr,
        )
        return 1

    # Step 2: build chelis-std as a reef package. The build emits
    # packages/chelis-std/dist/chelis-std-<version>.{tar.zst,chb}.
    print(
        f"[2/4] {chelis_bin.relative_to(repo)} reef build packages/chelis-std/",
        file=sys.stderr,
    )
    rc = subprocess.run(
        [str(chelis_bin), "reef", "build", "packages/chelis-std"],
        cwd=repo,
    ).returncode
    if rc != 0:
        print("ERROR: chelis reef build failed for packages/chelis-std", file=sys.stderr)
        return 1

    archive_src = pkg_dist / archive_name
    shell_src = pkg_dist / shell_name
    for src in (archive_src, shell_src):
        if not src.is_file():
            print(f"ERROR: expected build output missing: {src}", file=sys.stderr)
            return 1

    # Step 3: copy into the bundle crate's dist/ (overwriting whatever
    # was committed). The bundle crate's include_bytes!() pulls from
    # this exact filename, so the version constant in
    # crates/chelis-std-bundle/{build.rs,src/lib.rs} must agree with
    # the chelis-std reef.toml's version. If they ever diverge, fail
    # loudly here rather than silently embedding a wrong file.
    print(
        f"[3/4] copy artifacts into crates/chelis-std-bundle/dist/",
        file=sys.stderr,
    )
    for src in (archive_src, shell_src):
        dst = bundle_dist / src.name
        shutil.copy2(src, dst)

    # Step 4: show what changed under the bundle crate's dist/. Helps
    # the user confirm that their commit will pick up the right bytes.
    print("[4/4] git diff --stat crates/chelis-std-bundle/dist/", file=sys.stderr)
    subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            "--",
            str((bundle_dist).relative_to(repo)),
        ],
        cwd=repo,
    )

    print("OK: chelis-std bundle regenerated.", file=sys.stderr)
    print(
        "Next: `git add crates/chelis-std-bundle/dist/` and commit.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
