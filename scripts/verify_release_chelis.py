#!/usr/bin/env python3
"""Verify the portable Linux chelis toolchain Devenv release output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tomllib
from collections.abc import Sequence
from pathlib import Path


PUBLIC_RUNTIME_HEADERS = (
    "chelis_runtime.h",
    "chelis_runtime_dtype.h",
    "chelis_blas.h",
    "chelis_simd.h",
    "chelis_math.h",
)


class VerificationError(RuntimeError):
    """A portable release artifact violates its contract."""


def workspace_version(manifest_path: Path) -> str:
    """Read the exact workspace package version from Cargo.toml."""
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    try:
        version = manifest["workspace"]["package"]["version"]
    except (KeyError, TypeError) as error:
        raise VerificationError(
            "Cargo.toml has no workspace package version"
        ) from error
    if not isinstance(version, str) or not version:
        raise VerificationError("Cargo.toml has an invalid workspace package version")
    return version


def output_path_from_build_json(path: Path) -> Path:
    """Parse the release output path from Devenv build JSON."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        raw_output = document["outputs.release-chelis"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise VerificationError(f"cannot parse Devenv build result: {path}") from error
    if not isinstance(raw_output, str) or not raw_output.startswith("/"):
        raise VerificationError("Devenv build result has an invalid release output path")
    return Path(raw_output)


def asset_name(version: str) -> str:
    """Return the exact tarball name the chelisup installer requests."""
    return f"chelis-v{version}-linux-x86_64.tar.gz"


def expected_members(version: str) -> set[str]:
    """Return the exact member set of the release tarball."""
    stem = f"chelis-v{version}-linux-x86_64"
    members = {
        stem,
        f"{stem}/bin",
        f"{stem}/bin/chelis",
        f"{stem}/lib",
        f"{stem}/lib/libchelis_runtime.a",
        f"{stem}/include",
        f"{stem}/README.md",
        f"{stem}/LICENSE",
    }
    members.update(f"{stem}/include/{header}" for header in PUBLIC_RUNTIME_HEADERS)
    return members


def _verify_inventory(root: Path, version: str) -> tuple[Path, Path]:
    name = asset_name(version)
    expected = [name, f"{name}.sha256"]
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise VerificationError(f"cannot read release output: {root}: {error}") from error
    actual = [entry.name for entry in entries]
    if actual != expected:
        raise VerificationError(
            f"release output inventory differs: expected {expected}, got {actual}"
        )
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise VerificationError("release output inventory contains a non-regular file")
    return root / name, root / f"{name}.sha256"


def _verify_checksum(tarball: Path, checksum: Path) -> None:
    try:
        text = checksum.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise VerificationError(
            f"cannot read checksum file: {checksum}: {error}"
        ) from error
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", text)
    if match is None or match.group(2) != tarball.name:
        raise VerificationError("checksum file has an invalid sha256sum record")
    try:
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    except OSError as error:
        raise VerificationError(f"cannot hash release tarball: {error}") from error
    if digest != match.group(1):
        raise VerificationError("checksum does not match the release tarball")


def _verify_members(tarball: Path, version: str) -> None:
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            actual = {member.name.rstrip("/") for member in archive.getmembers()}
    except (OSError, tarfile.TarError) as error:
        raise VerificationError(f"cannot read release tarball: {error}") from error
    expected = expected_members(version)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise VerificationError(
            f"release tarball tree differs: missing {missing}, extra {extra}"
        )


def verify_tag_parity(version: str, ref_type: str | None, ref_name: str | None) -> None:
    """At a tag, the tag must equal v<workspace-version> before publication."""
    if ref_type != "tag":
        return
    expected = f"v{version}"
    if ref_name != expected:
        raise VerificationError(
            f"release tag differs from the workspace version: "
            f"expected {expected}, got {ref_name}"
        )


def verify_release_output(root: Path, version: str) -> None:
    """Verify the complete release output or raise VerificationError."""
    tarball, checksum = _verify_inventory(root, version)
    _verify_checksum(tarball, checksum)
    _verify_members(tarball, version)


def stage_release_output(source: Path, destination: Path, version: str) -> None:
    """Copy the exact release files into a new workspace directory."""
    tarball, checksum = _verify_inventory(source, version)
    try:
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(tarball, destination / tarball.name)
        shutil.copy2(checksum, destination / checksum.name)
    except OSError as error:
        raise VerificationError(
            f"cannot stage release output at {destination}: {error}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--build-json", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root or output_path_from_build_json(args.build_json)
        version = workspace_version(args.manifest)
        verify_tag_parity(
            version,
            os.environ.get("GITHUB_REF_TYPE"),
            os.environ.get("GITHUB_REF_NAME"),
        )
        verify_release_output(root, version)
        if args.stage_root is not None:
            stage_release_output(root, args.stage_root, version)
            root = args.stage_root
            verify_release_output(root, version)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"path={root}\n")
    except (OSError, VerificationError) as error:
        print(f"verify_release_chelis: {error}", file=sys.stderr)
        return 1
    print("verify_release_chelis: linux-x86_64 output is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
