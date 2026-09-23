#!/usr/bin/env python3
"""Regenerate the chelis-std bundle artifacts that ship inside the
chelis binary.

This script is the canonical pipeline for refreshing the bytes that
`crates/chelis-std-bundle` embeds via `include_bytes!()`. Run it from
the repo root after any change to `packages/chelis-std/src/**.ch`,
then commit all five owned outputs: the `.tar.zst`/`.chb` pair under
`packages/chelis-std/dist/`, the matching pair under
`crates/chelis-std-bundle/dist/`, and `packages/chelis-std/reef.lock`.

Pipeline:
  1. cargo build -p chelis --release
  2. (release-binary) chelis reef build packages/chelis-std/
     produces packages/chelis-std/dist/chelis-std-<version>.{tar.zst,chb}
  3. copy those into crates/chelis-std-bundle/dist/
  4. rebuild chelis so the binary embeds those final artifact bytes
  5. regenerate packages/chelis-std/reef.lock in a temporary source staging
     tree, retaining the lock but not the staging build's incidental artifacts
  6. git diff --stat to show what changed

Invocation:
  python3 scripts/regenerate_chelis_std_bundle.py [--debug] [--check]

Flags:
  --debug   build chelis in dev profile instead of release (faster
            iteration when wiring the script itself).
  --check   regenerate twice in isolated snapshots, require the committed
            five-output set to equal the first pass and the two passes to
            equal each other, then restore the exact committed bytes.

Exit codes:
  0  success (artifacts up-to-date or successfully regenerated)
  1  build/copy failed; stderr carries the underlying tool output.

The script is intentionally idempotent: re-running with no source
changes produces no diff. It pins SOURCE_DATE_EPOCH to 0 for every
subprocess so ambient build environments cannot change the committed
artifact identity. The committed artifacts ARE the source of truth.
The build.rs in chelis-std-bundle does not regenerate them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


CANONICAL_BUNDLE_SOURCE_DATE_EPOCH = "0"


def canonical_bundle_environment() -> dict[str, str]:
    """Return the inherited environment with the committed bundle epoch pinned."""
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = CANONICAL_BUNDLE_SOURCE_DATE_EPOCH
    return environment


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


def owned_generated_outputs(repo: Path, version: str) -> tuple[Path, ...]:
    """Every tracked output owned by this regeneration pipeline."""
    archive_name = f"chelis-std-{version}.tar.zst"
    shell_name = f"chelis-std-{version}.chb"
    return (
        repo / "packages" / "chelis-std" / "dist" / archive_name,
        repo / "packages" / "chelis-std" / "dist" / shell_name,
        repo / "crates" / "chelis-std-bundle" / "dist" / archive_name,
        repo / "crates" / "chelis-std-bundle" / "dist" / shell_name,
        repo / "packages" / "chelis-std" / "reef.lock",
    )


def stage_runtime_package_for_lock(package_root: Path, staged_root: Path) -> None:
    """Stage only the chelis-std inputs needed to regenerate its root lock.

    The final dist artifacts must not be overwritten after they have been
    copied into the compile-time bundle: the rebuilt CLI synthesizes the lock
    hashes from those exact embedded bytes. A temporary package build gives us
    that generated lock while its newly emitted dist directory remains
    disposable.
    """
    manifest_path = package_root / "reef.toml"
    with manifest_path.open("rb") as file:
        manifest = tomllib.load(file)
    source_roots = ["src", *manifest["package"].get("additional_sources", [])]

    staged_root.mkdir(parents=True)
    shutil.copy2(manifest_path, staged_root / "reef.toml")
    for source_root in source_roots:
        shutil.copytree(package_root / source_root, staged_root / source_root)


def regenerate_runtime_lock(
    chelis_bin: Path,
    package_root: Path,
    *,
    repository_root: Path,
) -> None:
    """Regenerate only ``package_root/reef.lock`` with the final embedded bundle."""
    with tempfile.TemporaryDirectory(prefix="chelis-std-lock-") as temp:
        staged_root = Path(temp) / "chelis-std"
        stage_runtime_package_for_lock(package_root, staged_root)
        rc = subprocess.run(
            [str(chelis_bin), "reef", "build", str(staged_root)],
            cwd=repository_root,
            env=canonical_bundle_environment(),
        ).returncode
        if rc != 0:
            raise RuntimeError("chelis reef build failed while regenerating chelis-std lock")
        staged_lock = staged_root / "reef.lock"
        if not staged_lock.is_file():
            raise RuntimeError(f"expected regenerated lock missing: {staged_lock}")
        shutil.copy2(staged_lock, package_root / "reef.lock")


def regenerate(repo: Path, *, debug: bool, show_diff: bool = True) -> int:
    """Run one complete canonical source-to-artifact generation pass."""
    environment = canonical_bundle_environment()
    version = chelis_std_version(repo)
    bundle_dist = repo / "crates" / "chelis-std-bundle" / "dist"
    pkg_dist = repo / "packages" / "chelis-std" / "dist"
    archive_name = f"chelis-std-{version}.tar.zst"
    shell_name = f"chelis-std-{version}.chb"
    owned_outputs = owned_generated_outputs(repo, version)

    bundle_dist.mkdir(parents=True, exist_ok=True)

    profile_flag = [] if debug else ["--release"]
    profile_dir = "debug" if debug else "release"

    # Step 1: build the chelis CLI.
    print(f"[1/6] cargo build -p chelis-cli {' '.join(profile_flag)}", file=sys.stderr)
    rc = subprocess.run(
        ["cargo", "build", "-p", "chelis-cli", *profile_flag],
        cwd=repo,
        env=environment,
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
        f"[2/6] {chelis_bin.relative_to(repo)} reef build packages/chelis-std/",
        file=sys.stderr,
    )
    rc = subprocess.run(
        [str(chelis_bin), "reef", "build", "packages/chelis-std"],
        cwd=repo,
        env=environment,
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
        f"[3/6] copy artifacts into crates/chelis-std-bundle/dist/",
        file=sys.stderr,
    )
    for src in (archive_src, shell_src):
        dst = bundle_dist / src.name
        shutil.copy2(src, dst)

    # Step 4: rebuild the CLI so its include_bytes! values are the final
    # artifacts just copied above. The root lock synthesized by the next step
    # must name these bytes, not the predecessor embedded by step 1.
    print(
        f"[4/6] rebuild chelis-cli with the refreshed embedded bundle",
        file=sys.stderr,
    )
    rc = subprocess.run(
        ["cargo", "build", "-p", "chelis-cli", *profile_flag],
        cwd=repo,
        env=environment,
    ).returncode
    if rc != 0:
        print("ERROR: cargo rebuild failed after bundle copy", file=sys.stderr)
        return 1

    # Step 5: generate the committed root lock from the now-current embedded
    # bytes without letting another package build replace the final dist pair.
    print("[5/6] regenerate packages/chelis-std/reef.lock", file=sys.stderr)
    try:
        regenerate_runtime_lock(chelis_bin, repo / "packages/chelis-std", repository_root=repo)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Step 6: show what changed under every owned generated surface. Helps
    # the user confirm that their commit will pick up the right bytes.
    if show_diff:
        print("[6/6] git diff --stat for owned chelis-std outputs", file=sys.stderr)
        subprocess.run(
            [
                "git",
                "diff",
                "--stat",
                "--",
                *(str(path.relative_to(repo)) for path in owned_outputs),
            ],
            cwd=repo,
            env=environment,
        )

    print("OK: chelis-std bundle regenerated.", file=sys.stderr)
    print(
        "Next: commit both dist pairs and packages/chelis-std/reef.lock.",
        file=sys.stderr,
    )
    return 0


def _snapshot_existing(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths if path.is_file()}


def _restore_snapshot(paths: tuple[Path, ...], snapshot: dict[Path, bytes]) -> None:
    for path in paths:
        if path in snapshot:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(snapshot[path])
        elif path.exists():
            path.unlink()


def snapshot_violations(committed, first, second):
    """Compare all owned outputs, including missing or unexpected identities."""
    paths = sorted(committed.keys() | first.keys() | second.keys())
    stale = [path for path in paths if committed.get(path) != first.get(path)]
    unstable = [path for path in paths if first.get(path) != second.get(path)]
    return stale, unstable


def check_generated_outputs(repo: Path, *, debug: bool) -> int:
    """Fail closed unless committed bytes are current and generation is stable."""
    version = chelis_std_version(repo)
    outputs = owned_generated_outputs(repo, version)
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        for path in missing:
            print(
                f"ERROR: missing generated output: {path.relative_to(repo)}",
                file=sys.stderr,
            )
        return 1

    committed = _snapshot_existing(outputs)
    first: dict[Path, bytes] = {}
    second: dict[Path, bytes] = {}
    try:
        print("[check 1/2] regenerate from committed inputs", file=sys.stderr)
        if regenerate(repo, debug=debug, show_diff=False) != 0:
            return 1
        first = _snapshot_existing(outputs)

        print("[check 2/2] regenerate again for byte determinism", file=sys.stderr)
        if regenerate(repo, debug=debug, show_diff=False) != 0:
            return 1
        second = _snapshot_existing(outputs)
    finally:
        _restore_snapshot(outputs, committed)

    stale, unstable = snapshot_violations(committed, first, second)
    for path in stale:
        print(
            f"ERROR: stale generated output: {path.relative_to(repo)}",
            file=sys.stderr,
        )
    for path in unstable:
        print(
            f"ERROR: nondeterministic generated output: {path.relative_to(repo)}",
            file=sys.stderr,
        )
    if stale or unstable:
        return 1

    print(
        "OK: all five chelis-std generated outputs are current and byte-stable.",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="build chelis in dev profile instead of release",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed outputs are current and byte-stable without changing them",
    )
    args = parser.parse_args()

    repo = repo_root()
    if args.check:
        return check_generated_outputs(repo, debug=args.debug)
    return regenerate(repo, debug=args.debug)


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        from ci_timing import subprocesses
        with subprocesses():
            sys.exit(main())
