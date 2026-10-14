#!/usr/bin/env python3
"""Regenerate Cargo.nix in an isolated workspace and compare exact bytes."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Protocol

_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_VERSION_LINE = re.compile(r"crate2nix ([0-9]+\.[0-9]+\.[0-9]+)")
_TIMEOUT = re.compile(r"[1-9][0-9]*")
_FEATURE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+./?:-]*")
_URL_USER_INFO = re.compile(r"(?P<prefix>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".devenv",
        ".direnv",
        ".git",
        ".venv",
        ".work",
        ".worktrees",
        "__pycache__",
        "node_modules",
        "target",
    }
)
_MAX_FEATURES = 64
_MAX_FEATURE_BYTES = 128
MAX_DIAGNOSTIC_BYTES = 4096


class Completed(Protocol):
    returncode: int
    stdout: str
    stderr: str


Run = Callable[[list[str], Path, Mapping[str, str], int], Completed]


class DriverError(ValueError):
    """A stable action failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


PROFILE_NAME = "check-cargo-nix"
_ACTIONS_ROOT = Path(__file__).resolve().parents[1]
if str(_ACTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACTIONS_ROOT))
from profile_contract import (  # noqa: E402
    ProfileRecord,
)
from profile_contract import (  # noqa: E402
    load_profile_record as _load_profile_record,
)
from profile_contract import (  # noqa: E402
    require_profile_match as _require_profile_match,
)


def load_profile_record(env: Mapping[str, str]) -> ProfileRecord:
    try:
        return _load_profile_record(
            env,
            profile=PROFILE_NAME,
            action="actions/check-cargo-nix",
            constructor="crate2nix-source",
            systems=("aarch64-darwin", "x86_64-linux"),
        )
    except ValueError as error:
        raise DriverError("profile", str(error)) from error


def require_profile_match(record: ProfileRecord, *, version: str) -> None:
    try:
        _require_profile_match(record, version=version)
    except ValueError as error:
        raise DriverError("profile", str(error)) from error


@dataclass(frozen=True)
class Version:
    value: str

    def __str__(self) -> str:
        return self.value


class FeatureMode(Enum):
    ALL = "all"
    DEFAULT = "default"
    NONE = "none"


@dataclass(frozen=True)
class Features:
    names: tuple[str, ...]


@dataclass(frozen=True)
class Timeout:
    seconds: int


@dataclass(frozen=True)
class WorkingDirectory:
    value: PurePosixPath


@dataclass(frozen=True)
class Inputs:
    version: Version
    feature_mode: FeatureMode
    features: Features
    timeout: Timeout
    working_directory: WorkingDirectory


@dataclass(frozen=True)
class ResolvedInputs:
    repository_root: Path
    working_directory: Path
    relative_working_directory: PurePosixPath


def parse_version(value: str) -> Version:
    if _VERSION.fullmatch(value) is None:
        raise DriverError(
            "cargo-nix-input",
            "crate2nix-version must be an exact X.Y.Z version",
        )
    return Version(value)


def parse_feature_mode(value: str) -> FeatureMode:
    try:
        return FeatureMode(value)
    except ValueError as error:
        raise DriverError(
            "cargo-nix-feature",
            "feature-mode must be exactly all, default, or none",
        ) from error


def parse_features(value: str, mode: FeatureMode) -> Features:
    if value == "":
        names: tuple[str, ...] = ()
    else:
        if "\x00" in value:
            raise DriverError(
                "cargo-nix-feature", "features contains an invalid character"
            )
        normalized = value.replace("\r\n", "\n")
        if "\r" in normalized:
            raise DriverError(
                "cargo-nix-feature", "features contains an invalid character"
            )
        candidates = normalized.split("\n")
        if candidates[-1] == "":
            candidates.pop()
        if any(candidate == "" for candidate in candidates):
            raise DriverError("cargo-nix-feature", "features contains a blank entry")
        if len(candidates) > _MAX_FEATURES:
            raise DriverError("cargo-nix-feature", "features exceeds the 64-name limit")
        if len(set(candidates)) != len(candidates):
            raise DriverError("cargo-nix-feature", "features contains a duplicate name")
        for candidate in candidates:
            if (
                len(candidate.encode("utf-8")) > _MAX_FEATURE_BYTES
                or _FEATURE.fullmatch(candidate) is None
            ):
                raise DriverError(
                    "cargo-nix-feature", "features contains an invalid name"
                )
        names = tuple(sorted(candidates, key=lambda name: name.encode("ascii")))
    if mode is FeatureMode.ALL and names:
        raise DriverError(
            "cargo-nix-feature", "feature-mode all does not accept explicit features"
        )
    return Features(names)


def parse_timeout(value: str) -> Timeout:
    if _TIMEOUT.fullmatch(value) is None:
        raise DriverError(
            "cargo-nix-input",
            "timeout-seconds must be an integer from 1 through 3600",
        )
    seconds = int(value)
    if seconds > 3600:
        raise DriverError(
            "cargo-nix-input",
            "timeout-seconds must be an integer from 1 through 3600",
        )
    return Timeout(seconds)


def parse_working_directory(value: str) -> WorkingDirectory:
    if not value or "\\" in value or "\x00" in value or "\r" in value or "\n" in value:
        raise DriverError("cargo-nix-input", "working-directory is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise DriverError(
            "cargo-nix-input", "working-directory must stay inside GITHUB_WORKSPACE"
        )
    normalized = PurePosixPath(".") if value == "." else path
    if any(part in {"", "."} for part in normalized.parts):
        raise DriverError("cargo-nix-input", "working-directory is invalid")
    return WorkingDirectory(normalized)


def parse_inputs(env: Mapping[str, str]) -> Inputs:
    version = parse_version(env.get("INPUT_CRATE2NIX_VERSION", ""))
    mode = parse_feature_mode(env.get("INPUT_FEATURE_MODE", "all"))
    features = parse_features(env.get("INPUT_FEATURES", ""), mode)
    timeout = parse_timeout(env.get("INPUT_TIMEOUT_SECONDS", "600"))
    working_directory = parse_working_directory(env.get("INPUT_WORKING_DIRECTORY", "."))
    return Inputs(version, mode, features, timeout, working_directory)


def _repository_relative(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return "<outside-workspace>"
    return relative or "."


def resolve_inputs(inputs: Inputs, env: Mapping[str, str]) -> ResolvedInputs:
    root_value = env.get("GITHUB_WORKSPACE", "")
    root = Path(root_value)
    if not root_value or not root.is_dir():
        raise DriverError("cargo-nix-input", "GITHUB_WORKSPACE is not a directory")
    try:
        repository_root = root.resolve(strict=True)
        selected = repository_root.joinpath(
            *inputs.working_directory.value.parts
        ).resolve(strict=True)
        selected.relative_to(repository_root)
    except (OSError, ValueError) as error:
        raise DriverError(
            "cargo-nix-input", "working-directory must stay inside GITHUB_WORKSPACE"
        ) from error
    if not selected.is_dir():
        raise DriverError("cargo-nix-input", "working-directory is not a directory")
    for name in ("Cargo.toml", "Cargo.lock", "Cargo.nix"):
        path = selected / name
        if not path.is_file():
            relative = _repository_relative(repository_root, path)
            raise DriverError(
                "cargo-nix-input", f"required file is missing: {relative}"
            )
    relative = selected.relative_to(repository_root)
    return ResolvedInputs(
        repository_root,
        selected,
        PurePosixPath(".")
        if not relative.parts
        else PurePosixPath(relative.as_posix()),
    )


def parse_runner_temp(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or "\x00" in value
        or "\r" in value
        or "\n" in value
        or not path.is_absolute()
        or not path.is_dir()
    ):
        raise DriverError(
            "cargo-nix-input", "RUNNER_TEMP must be an existing absolute directory"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise DriverError(
            "cargo-nix-input", "RUNNER_TEMP cannot be resolved"
        ) from error


def generation_argv(
    executable: str, mode: FeatureMode, features: tuple[str, ...]
) -> list[str]:
    argv = [executable, "generate"]
    if mode is FeatureMode.DEFAULT:
        argv.append("--default-features")
    elif mode is FeatureMode.NONE:
        argv.append("--no-default-features")
    for feature in features:
        argv.extend(("--features", feature))
    return argv


def _run(
    argv: list[str], cwd: Path, env: Mapping[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=dict(env),
        timeout=timeout,
    )


def verify_tool(
    expected: Version,
    cwd: Path,
    run: Run,
    env: Mapping[str, str],
) -> str:
    executable = "crate2nix"
    try:
        completed = run([executable, "--version"], cwd, env, 30)
    except (OSError, subprocess.SubprocessError) as error:
        raise DriverError(
            "cargo-nix-version-mismatch", "crate2nix version cannot be verified"
        ) from error
    first_line = (
        completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    )
    match = _VERSION_LINE.fullmatch(first_line)
    if completed.returncode != 0 or match is None or match.group(1) != str(expected):
        raise DriverError(
            "cargo-nix-version-mismatch",
            "installed crate2nix does not match crate2nix-version",
        )
    return executable


def _is_excluded(relative: PurePosixPath, path: Path) -> bool:
    if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return True
    if relative == PurePosixPath(".pre-commit-config.yaml") and path.is_symlink():
        return True
    return bool(
        relative.parts
        and len(relative.parts) == 1
        and path.is_symlink()
        and (relative.parts[0] == "result" or relative.parts[0].startswith("result-"))
    )


def _validate_link(path: Path, repository_root: Path) -> None:
    if not path.is_symlink():
        return
    target_text = os.readlink(path)
    target = Path(target_text)
    if target.is_absolute():
        raise DriverError(
            "cargo-nix-isolation", "an absolute symbolic link is not allowed"
        )
    resolved = (path.parent / target).resolve(strict=False)
    if not resolved.is_relative_to(repository_root):
        raise DriverError(
            "cargo-nix-isolation", "a symbolic link leaves GITHUB_WORKSPACE"
        )


def _validate_links(repository_root: Path, private_root: Path) -> None:
    try:
        for directory, directory_names, file_names in os.walk(
            repository_root, topdown=True, followlinks=False
        ):
            source_directory = Path(directory)
            if source_directory == private_root or source_directory.is_relative_to(
                private_root
            ):
                directory_names[:] = []
                continue

            kept_directories: list[str] = []
            for name in directory_names:
                path = source_directory / name
                if path == private_root or path.is_relative_to(private_root):
                    continue
                relative = PurePosixPath(path.relative_to(repository_root).as_posix())
                if _is_excluded(relative, path):
                    continue
                _validate_link(path, repository_root)
                if not path.is_symlink():
                    kept_directories.append(name)
            directory_names[:] = kept_directories

            for name in file_names:
                path = source_directory / name
                if path == private_root or path.is_relative_to(private_root):
                    continue
                relative = PurePosixPath(path.relative_to(repository_root).as_posix())
                if _is_excluded(relative, path):
                    continue
                _validate_link(path, repository_root)
    except DriverError:
        raise
    except OSError as error:
        raise DriverError(
            "cargo-nix-isolation", "workspace symbolic links cannot be inspected"
        ) from error


def _copy_workspace(
    repository_root: Path, destination: Path, private_root: Path
) -> None:
    _validate_links(repository_root, private_root)

    def ignore(directory: str, names: list[str]) -> set[str]:
        source_directory = Path(directory)
        ignored: set[str] = set()
        for name in names:
            candidate = source_directory / name
            if candidate == private_root or candidate.is_relative_to(private_root):
                ignored.add(name)
                continue
            relative = PurePosixPath(candidate.relative_to(repository_root).as_posix())
            if _is_excluded(relative, candidate):
                ignored.add(name)
        return ignored

    try:
        shutil.copytree(
            repository_root,
            destination,
            symlinks=True,
            ignore=ignore,
        )
    except (OSError, shutil.Error) as error:
        raise DriverError(
            "cargo-nix-isolation", "GITHUB_WORKSPACE cannot be copied"
        ) from error


def _sanitize_process_output(
    value: str,
    repository_root: Path,
    runner_temp: Path,
    temporary_root: Path,
) -> str:
    sanitized = value.replace(str(runner_temp), "<temporary>")
    sanitized = sanitized.replace(str(temporary_root), "<temporary>")
    sanitized = sanitized.replace(str(repository_root), "<workspace>")
    sanitized = _URL_USER_INFO.sub(r"\g<prefix><redacted>@", sanitized)
    lines = sanitized.splitlines()[-20:]
    text = "\n".join(lines)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > MAX_DIAGNOSTIC_BYTES:
        encoded = encoded[-MAX_DIAGNOSTIC_BYTES:]
        text = encoded.decode("utf-8", errors="replace")
    return text


def _repair_command(inputs: Inputs) -> str:
    return " ".join(
        generation_argv("crate2nix", inputs.feature_mode, inputs.features.names)
    )


def regenerate_and_compare(
    inputs: Inputs,
    resolved: ResolvedInputs,
    executable: str,
    runner_temp: Path,
    run: Run,
    env: Mapping[str, str],
) -> bytes:
    committed_path = resolved.working_directory / "Cargo.nix"
    try:
        committed = committed_path.read_bytes()
    except OSError as error:
        raise DriverError(
            "cargo-nix-input", "committed Cargo.nix cannot be read"
        ) from error

    try:
        with tempfile.TemporaryDirectory(
            prefix="check-cargo-nix-", dir=runner_temp
        ) as temporary_text:
            temporary_root = Path(temporary_text).resolve(strict=True)
            copied_root = temporary_root / "workspace"
            _copy_workspace(resolved.repository_root, copied_root, temporary_root)
            relative = resolved.relative_working_directory
            copied_workspace = (
                copied_root
                if relative == PurePosixPath(".")
                else copied_root.joinpath(*relative.parts)
            )
            generated_path = copied_workspace / "Cargo.nix"
            try:
                generated_path.unlink()
            except OSError as error:
                raise DriverError(
                    "cargo-nix-isolation", "copied Cargo.nix cannot be removed"
                ) from error

            argv = generation_argv(
                executable, inputs.feature_mode, inputs.features.names
            )
            try:
                completed = run(
                    argv,
                    copied_workspace,
                    env,
                    inputs.timeout.seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise DriverError(
                    "cargo-nix-timeout", "crate2nix generation exceeded timeout-seconds"
                ) from error
            except (OSError, subprocess.SubprocessError) as error:
                raise DriverError(
                    "cargo-nix-generation", "crate2nix generation cannot start"
                ) from error
            if completed.returncode != 0:
                tail = _sanitize_process_output(
                    completed.stdout + completed.stderr,
                    resolved.repository_root,
                    runner_temp,
                    temporary_root,
                )
                suffix = f"\n{tail}" if tail else ""
                raise DriverError(
                    "cargo-nix-generation",
                    f"crate2nix generation failed{suffix}",
                )
            if generated_path.is_symlink() or not generated_path.is_file():
                raise DriverError(
                    "cargo-nix-generation", "crate2nix did not create Cargo.nix"
                )
            try:
                generated = generated_path.read_bytes()
            except OSError as error:
                raise DriverError(
                    "cargo-nix-generation", "generated Cargo.nix cannot be read"
                ) from error
            if generated != committed:
                raise DriverError(
                    "cargo-nix-drift",
                    "committed Cargo.nix differs from exact regeneration. "
                    f"Run: {_repair_command(inputs)}",
                )
            return committed
    except DriverError:
        raise
    except OSError as error:
        raise DriverError(
            "cargo-nix-isolation", "the private workspace cannot be created"
        ) from error


def write_outputs(path: Path, version: Version, cargo_nix: bytes) -> None:
    digest = hashlib.sha256(cargo_nix).hexdigest()
    try:
        with path.open("a", encoding="utf-8") as output:
            output.write("cargo-nix-status=matched\n")
            output.write(f"crate2nix-version={version}\n")
            output.write(f"cargo-nix-sha256={digest}\n")
    except OSError as error:
        raise DriverError(
            "cargo-nix-input", "GitHub output values cannot be written"
        ) from error


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _emit_error(error: DriverError) -> None:
    print(
        f"::error title={error.code}::{_escape_data(str(error))}",
        file=sys.stderr,
    )


def main(
    *,
    env: Mapping[str, str] | None = None,
    run: Run = _run,
) -> int:
    selected_env = os.environ if env is None else env
    try:
        inputs = parse_inputs(selected_env)
        if "CHELIS_DEVENV_PROFILE" in selected_env:
            profile = load_profile_record(selected_env)
            require_profile_match(profile, version=str(inputs.version))
        resolved = resolve_inputs(inputs, selected_env)
        runner_temp = parse_runner_temp(selected_env.get("RUNNER_TEMP", ""))
        executable = verify_tool(
            inputs.version,
            resolved.working_directory,
            run,
            selected_env,
        )
        cargo_nix = regenerate_and_compare(
            inputs,
            resolved,
            executable,
            runner_temp,
            run,
            selected_env,
        )
        output_value = selected_env.get("GITHUB_OUTPUT", "")
        if output_value:
            write_outputs(Path(output_value), inputs.version, cargo_nix)
        return 0
    except DriverError as error:
        _emit_error(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
