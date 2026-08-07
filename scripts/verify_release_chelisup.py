#!/usr/bin/env python3
"""Verify one portable chelisup Devenv release output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path


SUPPORTED_PLATFORMS = frozenset({"linux-x86_64", "darwin-arm64"})
RunText = Callable[[list[str]], str]


class VerificationError(RuntimeError):
    """A portable release artifact violates its contract."""


def _default_run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or str(error)
        raise VerificationError(
            f"command failed: {' '.join(command)}: {detail}"
        ) from error
    except OSError as error:
        raise VerificationError(
            f"command failed: {' '.join(command)}: {error}"
        ) from error
    return completed.stdout


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
        raw_output = document["outputs.release-chelisup"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise VerificationError(f"cannot parse Devenv build result: {path}") from error
    if not isinstance(raw_output, str) or not raw_output.startswith("/"):
        raise VerificationError(
            "Devenv build result has an invalid release output path"
        )
    return Path(raw_output)


def _verify_inventory(root: Path, platform: str) -> tuple[Path, Path]:
    name = f"chelisup-{platform}"
    expected = [name, f"{name}.sha256"]
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise VerificationError(
            f"cannot read release output: {root}: {error}"
        ) from error
    actual = [entry.name for entry in entries]
    if actual != expected:
        raise VerificationError(
            f"release output inventory differs: expected {expected}, got {actual}"
        )
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise VerificationError("release output inventory contains a non-regular file")

    executable = root / name
    checksum = root / f"{name}.sha256"
    if not os.access(executable, os.X_OK):
        raise VerificationError(f"release executable is not executable: {executable}")
    return executable, checksum


def _verify_checksum(executable: Path, checksum: Path) -> None:
    try:
        text = checksum.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise VerificationError(
            f"cannot read checksum file: {checksum}: {error}"
        ) from error
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", text)
    if match is None or match.group(2) != executable.name:
        raise VerificationError("checksum file has an invalid sha256sum record")
    try:
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    except OSError as error:
        raise VerificationError(f"cannot hash release executable: {error}") from error
    if digest != match.group(1):
        raise VerificationError("checksum does not match the release executable")


def _verify_linux(executable: Path, run_text: RunText) -> None:
    description = run_text(["file", str(executable)])
    if "ELF" not in description or "x86-64" not in description:
        raise VerificationError("Linux release executable is not x86-64 ELF")
    program_headers = run_text(["readelf", "-l", str(executable)])
    dynamic_section = run_text(["readelf", "-d", str(executable)])
    if "INTERP" in program_headers or "NEEDED" in dynamic_section:
        raise VerificationError("Linux release executable is not fully static")


def _verify_darwin(executable: Path, run_text: RunText) -> None:
    description = run_text(["file", str(executable)])
    if "Mach-O" not in description or "arm64" not in description:
        raise VerificationError("Darwin release executable is not arm64 Mach-O")
    load_commands = run_text(["otool", "-L", str(executable)])
    if "/nix/store/" in "\n".join(load_commands.splitlines()[1:]):
        raise VerificationError("Darwin release dependency names the Nix store")
    for line in load_commands.splitlines()[1:]:
        fields = line.strip().split()
        if not fields:
            continue
        dependency = fields[0]
        if not dependency.startswith(("/usr/lib/", "/System/Library/Frameworks/")):
            raise VerificationError(
                f"Darwin release dependency is not an Apple system path: {dependency}"
            )


def stage_release_output(source: Path, destination: Path, platform: str) -> None:
    """Copy the exact platform files into a new workspace directory."""
    executable, checksum = _verify_inventory(source, platform)
    try:
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(executable, destination / executable.name)
        shutil.copy2(checksum, destination / checksum.name)
    except OSError as error:
        raise VerificationError(
            f"cannot stage release output at {destination}: {error}"
        ) from error


def verify_release_output(
    root: Path,
    platform: str,
    version: str,
    *,
    run_text: RunText = _default_run_text,
) -> None:
    """Verify the complete portable output or raise VerificationError."""
    if platform not in SUPPORTED_PLATFORMS:
        raise VerificationError(f"unsupported chelisup release platform: {platform}")
    executable, checksum = _verify_inventory(root, platform)
    _verify_checksum(executable, checksum)

    reported_version = run_text([str(executable), "--version"]).strip()
    expected_version = f"chelisup {version}"
    if reported_version != expected_version:
        raise VerificationError(
            f"release executable version differs: expected {expected_version}, got {reported_version}"
        )
    run_text([str(executable), "--help"])

    if platform == "linux-x86_64":
        _verify_linux(executable, run_text)
    else:
        _verify_darwin(executable, run_text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--build-json", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument(
        "--platform", choices=sorted(SUPPORTED_PLATFORMS), required=True
    )
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root or output_path_from_build_json(args.build_json)
        version = workspace_version(args.manifest)
        verify_release_output(root, args.platform, version)
        if args.stage_root is not None:
            stage_release_output(root, args.stage_root, args.platform)
            root = args.stage_root
            verify_release_output(root, args.platform, version)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with Path(github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"path={root}\n")
    except (OSError, VerificationError) as error:
        print(f"verify_release_chelisup: {error}", file=sys.stderr)
        return 1
    print(f"verify_release_chelisup: {args.platform} output is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
