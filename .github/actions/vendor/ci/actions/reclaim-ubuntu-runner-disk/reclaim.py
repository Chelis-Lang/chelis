#!/usr/bin/env python3
"""Reclaim disk on a verified GitHub-hosted Ubuntu runner."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

PURGE_PATHS: tuple[str, ...] = (
    "/usr/share/dotnet",
    "/usr/local/lib/android",
    "/opt/ghc",
    "/usr/local/.ghcup",
    "/usr/share/swift",
    "/usr/local/share/powershell",
    "/usr/local/share/boost",
    "/usr/lib/jvm",
    "/opt/hostedtoolcache/CodeQL",
)
PROTECTED_PATHS = frozenset(("/", "/usr", "/usr/local", "/opt", "/opt/hostedtoolcache"))
VERSION_ID = re.compile(r"^[0-9]{2}\.[0-9]{2}$")


class PolicyError(ValueError):
    """The internal cleanup policy is not safe."""


class ConfigurationError(ValueError):
    """The GitHub action environment is incomplete."""


class DiskUsage(Protocol):
    @property
    def free(self) -> int: ...


@dataclass(frozen=True)
class HostedUbuntu:
    """A parsed runner identity that permits the fixed cleanup plan."""

    version_id: str


@dataclass(frozen=True)
class PurgePlan:
    """Validated absolute cleanup targets."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class ReclaimResult:
    """Normalized public action result."""

    applied: bool
    failed_command_count: int
    reclaimed_bytes: int


def parse_runner(
    environ: Mapping[str, str], release: Mapping[str, str]
) -> HostedUbuntu | None:
    """Parse runner data into the only identity that permits deletion."""
    if environ.get("CI_RUNNER_OS") != "Linux":
        return None
    if environ.get("CI_RUNNER_ENVIRONMENT") != "github-hosted":
        return None
    if release.get("ID") != "ubuntu":
        return None
    version_id = release.get("VERSION_ID")
    if not isinstance(version_id, str) or VERSION_ID.fullmatch(version_id) is None:
        return None
    return HostedUbuntu(version_id)


def parse_purge_plan(paths: Sequence[str]) -> PurgePlan:
    """Parse reviewed constants and reject dangerous path drift."""
    parsed: list[str] = []
    for raw in paths:
        if not isinstance(raw, str):
            raise PolicyError("cleanup target is not text")
        path = PurePosixPath(raw)
        normalized = str(path)
        if not path.is_absolute() or normalized != raw:
            raise PolicyError("cleanup target is not a normalized absolute path")
        if normalized in PROTECTED_PATHS:
            raise PolicyError("cleanup target is a protected parent")
        if normalized in parsed:
            raise PolicyError("cleanup target is duplicated")
        parsed.append(normalized)
    if not parsed:
        raise PolicyError("cleanup plan is empty")
    return PurgePlan(tuple(parsed))


def read_disk_usage(path: str) -> DiskUsage:
    """Adapt the standard disk observer to the narrow cleanup contract."""
    return shutil.disk_usage(path)


def observe_free_bytes(disk_usage: Callable[[str], DiskUsage]) -> int | None:
    """Read root free capacity without changing cleanup status."""
    try:
        value = disk_usage("/").free
    except OSError:
        return None
    if not isinstance(value, int) or value < 0:
        return None
    return value


def run_best_effort(
    command: list[str],
    run_command: Callable[..., subprocess.CompletedProcess[object]],
) -> bool:
    """Run one fixed command and return whether it succeeded."""
    try:
        result = run_command(command, check=False)
    except OSError as error:
        print(
            f"runner-disk-reclaim:command-spawn-failed:{command[1]}:"
            f"{error.__class__.__name__}",
            file=sys.stderr,
        )
        return False
    print(f"runner-disk-reclaim:command:{command[1]}:exit:{result.returncode}")
    return result.returncode == 0


def reclaim_disk(
    environ: Mapping[str, str],
    *,
    os_release: Callable[[], Mapping[str, str]] = platform.freedesktop_os_release,
    run_command: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    disk_usage: Callable[[str], DiskUsage] = read_disk_usage,
) -> ReclaimResult:
    """Apply the fixed plan only after runner identity parsing succeeds."""
    plan = parse_purge_plan(PURGE_PATHS)
    try:
        release = os_release()
    except OSError:
        print("runner-disk-reclaim:skip:host-distribution-unavailable")
        return ReclaimResult(False, 0, 0)
    target = parse_runner(environ, release)
    if target is None:
        print("runner-disk-reclaim:skip:unsupported-runner")
        return ReclaimResult(False, 0, 0)

    print(f"runner-disk-reclaim:apply:ubuntu-{target.version_id}")
    before = observe_free_bytes(disk_usage)
    if before is not None:
        print(f"runner-disk-reclaim:free-bytes-before:{before}")

    failed = 0
    for path in plan.paths:
        if not run_best_effort(["sudo", "rm", "-rf", "--", path], run_command):
            failed += 1

    if which("docker") is not None and not run_best_effort(
        ["sudo", "docker", "image", "prune", "--all", "--force"],
        run_command,
    ):
        failed += 1

    reclaimed = 0
    if before is not None:
        after = observe_free_bytes(disk_usage)
        if after is not None:
            print(f"runner-disk-reclaim:free-bytes-after:{after}")
            reclaimed = max(0, after - before)
    print(f"runner-disk-reclaim:reclaimed-bytes:{reclaimed}")
    print(f"runner-disk-reclaim:failed-command-count:{failed}")
    return ReclaimResult(True, failed, reclaimed)


def write_outputs(result: ReclaimResult, environ: Mapping[str, str]) -> None:
    """Append the normalized result to the GitHub output file."""
    output_path = environ.get("GITHUB_OUTPUT")
    if not isinstance(output_path, str) or not output_path:
        raise ConfigurationError("GITHUB_OUTPUT is not set")
    text = (
        f"applied={'true' if result.applied else 'false'}\n"
        f"failed-command-count={result.failed_command_count}\n"
        f"reclaimed-bytes={result.reclaimed_bytes}\n"
    )
    with Path(output_path).open("a", encoding="utf-8") as output:
        output.write(text)


def main() -> int:
    """Run the action adapter."""
    try:
        result = reclaim_disk(os.environ)
        write_outputs(result, os.environ)
    except (ConfigurationError, PolicyError, OSError) as error:
        print(
            f"runner-disk-reclaim:error:{error.__class__.__name__}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
