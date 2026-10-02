#!/usr/bin/env python3
"""Job-local cache lifecycle; Tunnet host membership, never inputs, is authority.

The roots manifest is UTF-8, one canonical realized /nix/store output per line.
It is producer-owned: no store enumeration, registration-time heuristic, probe
build, or synthetic artifact participates in real-workload evidence.

Kache config/CLI/layout contracts were reviewed in kunobi-ninja/kache v0.12.0
and v0.16.0 (Chelis source a21d020142b1248537cd548ccde99a04c0a44820):
src/config.rs, daemon.rs, remote_backend.rs, remote_layout.rs, and main.rs. Both use v3
packs/manifests and static KACHE_S3_* signing placeholders before providers.
The supplied binary retains its own cache-key schema (including Chelis patches).
Only S3 over the fixed mesh endpoint is supported; no filesystem backend,
planner, credential broker, host service installation, or backend auto-detection.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any, NamedTuple

import warm_cache_write as kache
import warm_nix_write as nix

SCHEMA = "chelis-build-cache/v1"
ROOT_NAME = re.compile(r"bc-[a-z0-9_]{8}")
# Linux binds at most 108 bytes of Unix socket path, including the terminator.
SOCKET_PATH_LIMIT = 107
VERSIONS = ("0.12.0", "0.16.0")
CRATE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}")
MAX_DOCUMENT = 256 * 1024
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_ROOTS = 400
MAX_KEYS = 2000
CLEANUP_SECONDS = 30
PUBLISH_SECONDS = 1200
DEADLINE: ContextVar[float | None] = ContextVar("build_cache_deadline", default=None)
CLEANING_UP: ContextVar[bool] = ContextVar("build_cache_cleaning_up", default=False)
IDENTITY = ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB")
EXPORTED = frozenset(
    (
        "BUILD_CACHE_ROOTS_PATH",
        "NIX_CONFIG",
        "KACHE_CONFIG",
        "RUSTC_WRAPPER",
        "KACHE_CACHE_DIR",
        "KACHE_RUNTIME_DIR",
        "KACHE_SOCKET_PATH",
        "KACHE_S3_BUCKET",
        "KACHE_S3_ENDPOINT",
        "KACHE_S3_REGION",
        "KACHE_S3_PREFIX",
        "KACHE_S3_PROFILE",
        "KACHE_S3_ACCESS_KEY",
        "KACHE_S3_SECRET_KEY",
        "KACHE_LOCAL_ONLY",
        "KACHE_REMOTE_READONLY",
        "KACHE_DAEMON_IDLE_TIMEOUT",
    )
)


class CacheError(ValueError):
    """A bounded diagnostic; never include arbitrary state or child output."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CacheError("build-cache:" + code)


# Credential-shaped tokens never reach the job log, even from a child's output.
SECRET_SHAPES = re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}|[A-Za-z0-9+/_-]{40,}")


FAILURE_LINE = re.compile(r"error|warn|fail|denied|timed? ?out|refused", re.IGNORECASE)


def diagnostic(code: str, output: str) -> None:
    """Bounded, redacted child output for a failed command: the lines that
    name a failure, then the tail. The error itself stays a fixed code;
    these lines are what an operator reads. Progress lines are dropped, so a
    transfer's cause survives its own progress output."""
    lines = [line.strip() for line in output.replace("\r", "\n").splitlines()]
    lines = [line for line in lines if line and not line.startswith("Uploading: ")]
    failures = [line for line in lines if FAILURE_LINE.search(line)][:20]
    shown = failures + [line for line in lines[-4:] if line not in failures]
    for line in shown:
        print(
            "build-cache:" + code + ": " + SECRET_SHAPES.sub("<redacted>", line)[:400],
            file=sys.stderr,
        )


def remaining(limit: int) -> float:
    deadline = DEADLINE.get()
    seconds = (
        float(limit) if deadline is None else min(limit, deadline - time.monotonic())
    )
    require(seconds > 0, "publish-deadline")
    return seconds


def bounded_text(path: Path, limit: int = MAX_DOCUMENT) -> str:
    require(path.is_file() and not path.is_symlink(), "file-unsafe")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    require(len(data) <= limit, "file-too-large")
    return data.decode("utf-8")


def text_value(value: object) -> str:
    require(
        isinstance(value, str) and len(value) <= MAX_DOCUMENT and "\0" not in value,
        "value-invalid",
    )
    assert isinstance(value, str)
    return value


def workspace_path(value: str, caller: dict[str, str]) -> Path:
    workspace = Path(caller["GITHUB_WORKSPACE"]).resolve(strict=True)
    relative = Path(value)
    require(
        not relative.is_absolute() and ".." not in relative.parts, "workspace-unsafe"
    )
    selected = (workspace / relative).resolve(strict=True)
    require(
        selected.is_dir() and selected.is_relative_to(workspace), "workspace-unsafe"
    )
    return selected


def job_identity(caller: dict[str, str]) -> dict[str, str]:
    result = {name: caller.get(name, "") for name in IDENTITY}
    require(all(result.values()), "job-identity-missing")
    return result


def owned_root(state_path: str, caller: dict[str, str]) -> Path:
    temporary = Path(caller["RUNNER_TEMP"]).resolve(strict=True)
    path = Path(state_path)
    require(path.is_absolute() and path.name == "state.json", "state-path")
    root = path.parent
    require(
        root.parent == temporary and ROOT_NAME.fullmatch(root.name) is not None,
        "state-path",
    )
    require(not root.is_symlink() and root.is_dir(), "state-root")
    info = root.stat()
    require(
        info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
        "state-owner",
    )
    require(
        json.loads(bounded_text(root / "owner.json")) == job_identity(caller),
        "root-job-owner",
    )
    return root


def read_state(root: Path, caller: dict[str, str]) -> dict[str, Any]:
    value = json.loads(bounded_text(root / "state.json"))
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema",
            "identity",
            "workspace",
            "mode",
            "binary",
            "version",
            "previous",
            "environment",
        },
        "state-shape",
    )
    require(
        value["schema"] == SCHEMA and value["identity"] == job_identity(caller),
        "state-identity",
    )
    require(value["mode"] in ("read", "read-write"), "state-mode")
    workspace = Path(text_value(value["workspace"]))
    require(
        workspace.is_absolute()
        and workspace.is_dir()
        and workspace.is_relative_to(Path(caller["GITHUB_WORKSPACE"]).resolve()),
        "state-workspace",
    )
    binary, version = (text_value(value[key]) for key in ("binary", "version"))
    require(
        (not binary and not version)
        or (Path(binary).is_absolute() and version in VERSIONS),
        "state-kache",
    )
    for field in ("previous", "environment"):
        mapping = value[field]
        require(
            isinstance(mapping, dict) and set(mapping) <= EXPORTED, "state-environment"
        )
        for entry in mapping.values():
            text_value(entry)
    require(set(value["previous"]) == set(value["environment"]), "state-environment")
    expected = exposure(
        root, caller, binary, version, value["previous"].get("NIX_CONFIG", "")
    )
    require(value["environment"] == expected, "state-environment")
    return value


def toml_value(value: object) -> str:
    """Serialize parsed TOML values without discarding repository compiler settings."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                json.dumps(key) + " = " + toml_value(item)
                for key, item in value.items()
            )
            + " }"
        )
    raise CacheError("build-cache:unsupported-config-value")


def repository_config(workspace: Path, caller: dict[str, str]) -> dict[str, Any]:
    explicit = caller.get("KACHE_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = workspace / path
        return tomllib.loads(bounded_text(path))
    # Match Kache's nearest-project precedence; do not adopt machine user config.
    for directory in (workspace, *workspace.parents):
        candidate = directory / ".kache.toml"
        if candidate.exists():
            return tomllib.loads(bounded_text(candidate))
    return {}


def isolated_config(
    original: dict[str, Any], root: Path, version: str, readonly: bool
) -> str:
    require(version in VERSIONS, "unsupported-kache-version")
    config = copy.deepcopy(original)
    cache = config.setdefault("cache", {})
    require(isinstance(cache, dict), "config-cache")
    # Planner endpoints are a separate authority and are not part of this API.
    require(
        not config.get("planner") and not cache.get("planner"), "unsupported-planner"
    )
    cache.update(
        {
            "local_store": str(root / "kache"),
            "local_only": False,
            "remote_readonly": readonly,
            "daemon_idle_timeout_secs": 60,
            # A burst of new connections through the mesh occasionally loses a
            # SYN, and the client's 3-second connect deadline has no retry.
            "s3_concurrency": 4,
        }
    )
    if version == "0.16.0":
        cache["runtime_dir"] = str(root / "runtime")
    cache["remote"] = {
        "type": "s3",
        "bucket": kache.BUCKET,
        "region": kache.REGION,
        "endpoint": kache.ENDPOINT,
        "prefix": "",
    }
    # The reviewed gateway and IAM policy admit only root-level v3/_manifests.
    # Compiler cache keys already bind inputs, toolchain, and schema version.
    return (
        "\n".join(
            json.dumps(key) + " = " + toml_value(value) for key, value in config.items()
        )
        + "\n"
    )


def exposure(
    root: Path, caller: dict[str, str], binary: str, version: str, original_nix: str
) -> dict[str, str]:
    del caller
    result = {
        "BUILD_CACHE_ROOTS_PATH": str(root / "roots.txt"),
        "NIX_CONFIG": original_nix.rstrip()
        + "\nextra-substituters = "
        + nix.SERVER_URL
        + "\nextra-trusted-public-keys = "
        + nix.PUBLIC_KEY
        + "\nrequire-sigs = true\n",
    }
    if binary:
        result.update(
            {
                "RUSTC_WRAPPER": binary,
                "KACHE_CONFIG": str(root / "kache.toml"),
                "KACHE_CACHE_DIR": str(root / "kache"),
                "KACHE_S3_BUCKET": kache.BUCKET,
                "KACHE_S3_REGION": kache.REGION,
                "KACHE_S3_ENDPOINT": kache.ENDPOINT,
                "KACHE_S3_PREFIX": "",
                "KACHE_S3_PROFILE": "",
                "KACHE_S3_ACCESS_KEY": kache.PUBLIC_SIGNING_VALUE,
                "KACHE_S3_SECRET_KEY": kache.PUBLIC_SIGNING_VALUE,
                "KACHE_LOCAL_ONLY": "false",
                "KACHE_REMOTE_READONLY": "true",
                "KACHE_DAEMON_IDLE_TIMEOUT": "60",
            }
        )
        if version == "0.16.0":
            result.update(
                {
                    "KACHE_RUNTIME_DIR": str(root / "runtime"),
                    "KACHE_SOCKET_PATH": str(socket_path(root, version)),
                }
            )
    return result


def attempt(
    argv: list[str],
    environment: dict[str, str],
    cwd: Path,
    code: str,
    timeout: int = 300,
) -> tuple[int, str]:
    """Run a bounded child and return its exit status with its output."""
    try:
        result = subprocess.run(
            argv,
            env=environment,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=remaining(timeout),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise CacheError("build-cache:" + code + "-unavailable-or-timeout") from None
    return result.returncode, result.stdout + (
        "\n" + result.stderr if result.stderr else ""
    )


def command(
    argv: list[str],
    environment: dict[str, str],
    cwd: Path,
    code: str,
    timeout: int = 300,
) -> str:
    status, output = attempt(argv, environment, cwd, code, timeout)
    if status != 0:
        diagnostic(code, output)
        raise CacheError("build-cache:" + code)
    return output


def kache_version(binary: str, environment: dict[str, str], cwd: Path) -> str:
    require(Path(binary).is_absolute() and os.access(binary, os.X_OK), "kache-binary")
    version = command(
        [binary, "--version"], environment, cwd, "kache-version", 30
    ).strip()
    require(
        version in tuple("kache " + item for item in VERSIONS),
        "unsupported-kache-version",
    )
    return version.removeprefix("kache ")


def github_values(
    caller: dict[str, str], variable: str, values: dict[str, str]
) -> None:
    target = Path(caller[variable])
    temporary = Path(caller["RUNNER_TEMP"]).resolve(strict=True)
    require(
        target.is_absolute()
        and not target.is_symlink()
        and target.resolve(strict=True).is_relative_to(temporary),
        "github-file",
    )
    with target.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            text_value(value)
            marker = "cache_" + os.urandom(16).hex()
            require(marker not in value.splitlines(), "github-delimiter")
            handle.write(f"{key}<<{marker}\n{value}\n{marker}\n")


def remove_root(root: Path) -> None:
    # Separate bounded process: large caches cannot hold the runner's post-job
    # step indefinitely. A hard kill still relies on host-enforced cleanup.
    command(
        [
            sys.executable,
            "-I",
            "-c",
            "import shutil,sys; shutil.rmtree(sys.argv[1])",
            str(root),
        ],
        dict(os.environ),
        root.parent,
        "cleanup",
        CLEANUP_SECONDS,
    )


@contextmanager
def cleanup_deadline() -> Iterator[None]:
    token = DEADLINE.set(time.monotonic() + CLEANUP_SECONDS)
    cleaning = CLEANING_UP.set(True)
    try:
        yield
    finally:
        CLEANING_UP.reset(cleaning)
        DEADLINE.reset(token)


def socket_path(root: Path, version: str) -> Path:
    """The daemon socket for a store root. Every path component below the
    job workspace is short because the kernel refuses a longer socket path."""
    path = root / ("runtime" if version == "0.16.0" else "kache") / "daemon.sock"
    require(len(str(path).encode()) <= SOCKET_PATH_LIMIT, "kache-socket-path")
    return path


def daemon_running(root: Path, version: str) -> bool:
    validate_kache_paths(root)
    endpoint = socket_path(root, version)
    coordinator = endpoint.with_suffix(".state.json")
    lock = endpoint.with_suffix(".run.lock")
    require(
        not any(path.is_symlink() for path in (endpoint, coordinator, lock)),
        "kache-store-unsafe",
    )
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        require(
            not endpoint.exists() and not coordinator.exists(),
            "kache-run-lock-missing",
        )
        return False
    with os.fdopen(descriptor, "wb") as handle:
        info = os.fstat(handle.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
            "kache-store-unsafe",
        )
        try:
            # Both reviewed versions hold flock through runtime/worker teardown.
            # Socket/coordinator removal and the shutdown ACK happen earlier.
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def shutdown_daemon(
    binary: str, version: str, root: Path, environment: dict[str, str], code: str
) -> None:
    token = DEADLINE.set(time.monotonic() + remaining(CLEANUP_SECONDS))
    try:
        if not daemon_running(root, version):
            return
        # An idle exit can race the CLI request. Only lock release proves
        # success; a failed request alone never authorizes deletion.
        with suppress(CacheError):
            command(
                [binary, "daemon", "stop"], environment, root, code, CLEANUP_SECONDS
            )
        while daemon_running(root, version):
            time.sleep(min(0.05, remaining(CLEANUP_SECONDS)))
    finally:
        DEADLINE.reset(token)


def stop_kache(state: dict[str, Any], root: Path, caller: dict[str, str]) -> None:
    if state["binary"]:
        validate_kache_paths(root)
        original = tomllib.loads(bounded_text(root / "kache.toml"))
        safe = isolated_config(original, root, state["version"], True)
        (root / "kache.toml").write_text(safe, encoding="utf-8")
        environment = dict(caller) | exposure(
            root, caller, state["binary"], state["version"], ""
        )
        shutdown_daemon(
            state["binary"], state["version"], root, environment, "daemon-stop"
        )


def cleanup(state_path: str, caller: dict[str, str]) -> None:
    root = owned_root(state_path, caller)
    with cleanup_deadline():
        state = None
        try:
            state = read_state(root, caller)
            stop_kache(state, root, caller)
        finally:
            try:
                if state is not None and caller.get("GITHUB_ENV"):
                    github_values(caller, "GITHUB_ENV", state["previous"])
            finally:
                # Invalid state cannot authorize a binary, nor can a failed
                # stop authorize deletion beneath a still-running worker.
                for version in VERSIONS:
                    require(not daemon_running(root, version), "daemon-still-running")
                remove_root(root)


def write_state(root: Path, state: dict[str, Any]) -> None:
    """Replace owned lifecycle state without exposing a partial transition."""
    descriptor, name = tempfile.mkstemp(prefix=".state-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream)
        os.replace(name, root / "state.json")
    finally:
        Path(name).unlink(missing_ok=True)


def repository_kache(
    caller: dict[str, str], selected: Path, binary: str
) -> tuple[str, str]:
    require(
        not binary
        or not any(
            caller.get(key) for key in caller if key.startswith("KACHE_PLANNER_")
        ),
        "unsupported-planner",
    )
    if binary and not Path(binary).is_absolute():
        binary = str((selected / binary).resolve(strict=True))
    version = kache_version(binary, caller, selected) if binary else ""
    if binary and caller.get("RUSTC_WRAPPER"):
        require(
            Path(caller["RUSTC_WRAPPER"]).resolve() == Path(binary).resolve(),
            "repository-wrapper-conflict",
        )
    return binary, version


def restore(
    caller: dict[str, str], workspace: str = ".", mode: str = "read", binary: str = ""
) -> dict[str, str]:
    require(mode in ("read", "read-write"), "mode-invalid")
    identity = job_identity(caller)
    selected = workspace_path(workspace, caller)
    binary, version = repository_kache(caller, selected, binary)
    temporary = Path(caller["RUNNER_TEMP"]).resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="bc-", dir=temporary))
    try:
        (root / "owner.json").write_text(json.dumps(identity), encoding="utf-8")
        (root / "owner.json").chmod(0o600)
        environment = exposure(
            root, caller, binary, version, caller.get("NIX_CONFIG", "")
        )
        state = {
            "schema": SCHEMA,
            "identity": identity,
            "workspace": str(selected),
            "mode": mode,
            "binary": binary,
            "version": version,
            "environment": environment,
            "previous": {name: caller.get(name, "") for name in environment},
        }
        if binary:
            original = repository_config(selected, caller)
            (root / "kache.toml").write_text(
                isolated_config(original, root, version, True), encoding="utf-8"
            )
        (root / "roots.txt").touch(mode=0o600)
        write_state(root, state)
        if binary:
            command(
                [binary, "daemon", "start"],
                dict(caller) | environment,
                root,
                "daemon-start",
                30,
            )
        outputs = {
            "state-path": str(root / "state.json"),
            "roots-path": str(root / "roots.txt"),
        }
        # Restore is demand-driven: ordinary Nix substitution and Kache's exact
        # compiler-key reads. No Cargo metadata command rewrites Cargo.lock and
        # no imperative Cargo result is confused with a Nix derivation output.
        if caller.get("GITHUB_OUTPUT"):
            github_values(caller, "GITHUB_OUTPUT", outputs)
        if caller.get("GITHUB_ENV"):
            github_values(caller, "GITHUB_ENV", environment)
        return outputs
    except BaseException:
        if (root / "state.json").is_file():
            cleanup(str(root / "state.json"), caller)
        else:
            remove_root(root)
        raise


def attach(state_path: str, caller: dict[str, str], binary: str) -> dict[str, str]:
    """Attach the realized project wrapper to an existing Nix-only lifecycle."""
    root = owned_root(state_path, caller)
    original = read_state(root, caller)
    require(not original["binary"], "kache-already-attached")
    require(bool(binary), "attach-kache-binary")
    require(
        caller.get("NIX_CONFIG", "") == original["environment"]["NIX_CONFIG"],
        "nix-environment-changed",
    )
    selected = Path(original["workspace"])
    binary, version = repository_kache(caller, selected, binary)
    environment = exposure(
        root, caller, binary, version, original["previous"].get("NIX_CONFIG", "")
    )
    state = original | {
        "binary": binary,
        "version": version,
        "environment": environment,
        # Nix retains its pre-restore values; Kache restores the newly activated
        # project's values, not the portable bootstrap environment.
        "previous": {
            name: original["previous"].get(name, caller.get(name, ""))
            for name in environment
        },
    }
    (root / "kache.toml").write_text(
        isolated_config(repository_config(selected, caller), root, version, True),
        encoding="utf-8",
    )
    write_state(root, state)
    try:
        command(
            [binary, "daemon", "start"],
            dict(caller) | environment,
            root,
            "daemon-start",
            30,
        )
        outputs = {
            "state-path": str(root / "state.json"),
            "roots-path": str(root / "roots.txt"),
        }
        if caller.get("GITHUB_OUTPUT"):
            github_values(caller, "GITHUB_OUTPUT", outputs)
        if caller.get("GITHUB_ENV"):
            github_values(caller, "GITHUB_ENV", environment)
        return outputs
    except BaseException:
        # The Nix restore already succeeded in an earlier Actions step. Keep
        # that lifecycle usable by its always() finalizer after attachment fails.
        # If stopping fails, retain the new state so cleanup still owns the daemon.
        with cleanup_deadline():
            stop_kache(state, root, caller)
            write_state(root, original)
        raise


def declared_roots(root: Path) -> tuple[str, ...]:
    lines = bounded_text(root / "roots.txt").splitlines()
    require(len(lines) <= MAX_ROOTS, "too-many-roots")
    require(
        all(
            nix.STORE_PATH.fullmatch(path) is not None and not path.endswith(".drv")
            for path in lines
        ),
        "root-invalid",
    )
    return tuple(dict.fromkeys(lines))


def transport_environment(caller: dict[str, str]) -> dict[str, str]:
    environment = {
        key: caller[key]
        for key in ("PATH", "HOME", "SSL_CERT_FILE", "NIX_SSL_CERT_FILE")
        if key in caller
    }
    environment["NIX_CONFIG"] = (
        "experimental-features = nix-command\ntrusted-public-keys = "
        + nix.PUBLIC_KEY
        + "\nrequire-sigs = true\n"
    )
    environment["NIX_USER_CONF_FILES"] = "/dev/null"
    return environment


def nix_records(
    paths: tuple[str, ...], root: Path, environment: dict[str, str], store: str = ""
) -> tuple[nix.PathRecord, ...]:
    argv = ["nix", "path-info", "--json"]
    if store:
        argv += ["--store", store]
    # JSON stdout is separate from stderr; diagnostics must never become JSON.
    try:
        result = subprocess.run(
            [*argv, *paths],
            env=environment,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=remaining(120),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise CacheError("build-cache:path-info-unavailable-or-timeout") from None
    require(result.returncode == 0, "roots-not-realized")
    records = nix.parse_path_info(result.stdout)
    require({record.path for record in records} == set(paths), "root-records")
    return records


def download_archive(url: str, archive: Path) -> str:
    # A socket timeout is only an inactivity bound: even read1 can spend an
    # unbounded time parsing trickled HTTP headers/chunks. The child bounds the
    # entire transfer and is killed/reaped on timeout or cancellation.
    return command(
        [
            sys.executable,
            "-I",
            "-c",
            "\n".join(
                (
                    "import hashlib, sys, urllib.request",
                    "digest = hashlib.sha256()",
                    "total = 0",
                    "with urllib.request.urlopen(sys.argv[1], timeout=30) as response, open(sys.argv[2], 'wb') as output:",
                    "    while block := response.read(64 * 1024):",
                    "        total += len(block)",
                    "        if total > int(sys.argv[3]):",
                    "            raise ValueError('archive too large')",
                    "        digest.update(block)",
                    "        output.write(block)",
                    "print(digest.hexdigest())",
                )
            ),
            url,
            str(archive),
            str(MAX_ARCHIVE),
        ],
        dict(os.environ),
        archive.parent,
        "niks3-download",
        PUBLISH_SECONDS,
    ).strip()


def stage_niks3(root: Path) -> Path:
    require(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "unsupported-niks3-platform",
    )
    package = nix.load_package(
        (Path(__file__).parent / nix.INVENTORY.relative_to("tools")).read_text(
            encoding="utf-8"
        )
    )
    archive = root / "niks3.tar.gz"
    # No nix-prefetch-url: setup/publication must not register artifacts in the
    # production store. Only the reviewed digest's one regular member is staged.
    require(download_archive(package.url, archive) == package.sha256, "niks3-digest")
    binary = root / "niks3"
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember("niks3")
        require(member.isfile() and member.size <= MAX_ARCHIVE, "niks3-member")
        source = handle.extractfile(member)
        require(source is not None, "niks3-member")
        assert source is not None
        with source, binary.open("wb") as output:
            shutil.copyfileobj(source, output)
    binary.chmod(0o700)
    return binary


def verify_nix_records(
    before: tuple[nix.PathRecord, ...], after: tuple[nix.PathRecord, ...]
) -> None:
    require(
        {record.path for record in before} == {record.path for record in after},
        "nix-readback-incomplete",
    )
    expected = {record.path: record.nar_hash for record in before}
    for record in after:
        require(record.nar_hash == expected[record.path], "nix-readback-altered")
        require(
            nix.SIGNING_KEY in nix.signature_keys(record.signatures),
            "nix-readback-unsigned",
        )


def publish_nix(
    paths: tuple[str, ...], root: Path, caller: dict[str, str], evidence: dict[str, Any]
) -> None:
    if not paths:
        return
    environment = transport_environment(caller)
    before = nix_records(paths, root, environment)
    binary = stage_niks3(root)
    evidence["attempted_roots"] = len(paths)
    command(
        [
            str(binary),
            "push",
            "--server-url",
            nix.SERVER_URL,
            "--auth-token",
            nix.ANONYMOUS_TOKEN,
            *paths,
        ],
        environment,
        root,
        "nix-push",
        900,
    )
    # Readback verifies each declared root against the cache itself: its
    # narinfo (NAR hash and the fixed signing key) and its NAR bytes. It does
    # not copy the root's closure back: a toolchain closure is gigabytes, and
    # the closure members carry their own signatures for every consumer.
    for record in before:
        after = nix_records((record.path,), root, environment, nix.SERVER_URL)
        verify_nix_records((record,), after)
        command(
            [
                "nix",
                "store",
                "verify",
                "--store",
                nix.SERVER_URL,
                "--sigs-needed",
                "1",
                "--option",
                "trusted-public-keys",
                nix.PUBLIC_KEY,
                record.path,
            ],
            environment,
            root,
            "nix-integrity",
            300,
        )
        evidence["verified_roots"] += 1


def validate_kache_paths(root: Path) -> None:
    for relative in ("kache", "kache/store", "kache/store/blobs", "runtime"):
        path = root / relative
        require(
            not path.is_symlink() and path.resolve().is_relative_to(root),
            "kache-store-unsafe",
        )


class EntrySnapshot(NamedTuple):
    crate_name: str
    artifacts: int


def entry_snapshot(root: Path) -> dict[str, EntrySnapshot]:
    """Validate every entry's metadata and artifact files in both reviewed
    blob-store versions. Blob bytes are content-addressed and Kache verifies
    their hash at the download trust boundary, so they are not re-hashed
    here."""
    store = root / "store"
    require(not root.is_symlink() and not store.is_symlink(), "kache-store-unsafe")
    if not store.exists():
        return {}
    require(store.is_dir(), "kache-store-unsafe")
    keys = tuple(
        sorted(
            item.name
            for item in store.iterdir()
            if kache.CACHE_KEY.fullmatch(item.name)
        )
    )
    require(len(keys) <= MAX_KEYS, "kache-keys")
    digests: dict[str, EntrySnapshot] = {}
    for key in keys:
        remaining(300)
        directory = store / key
        require(directory.is_dir() and not directory.is_symlink(), "kache-entry-unsafe")
        meta = json.loads(bounded_text(directory / "meta.json"))
        require(isinstance(meta, dict) and meta.get("cache_key") == key, "kache-meta")
        crate_name = meta.get("crate_name")
        require(
            isinstance(crate_name, str)
            and CRATE_NAME.fullmatch(crate_name) is not None,
            "kache-crate",
        )
        files = meta.get("files")
        require(isinstance(files, list) and 0 < len(files) <= 10000, "kache-meta")
        names: set[str] = set()
        for entry in files:
            require(isinstance(entry, dict), "kache-meta")
            name, blob_hash, size = (
                entry.get("name"),
                entry.get("hash"),
                entry.get("size"),
            )
            require(
                isinstance(name, str)
                and name not in names
                and name not in ("", ".", "..")
                and "/" not in name
                and "\\" not in name
                and "\0" not in name,
                "kache-artifact-name",
            )
            require(
                isinstance(blob_hash, str)
                and kache.CACHE_KEY.fullmatch(blob_hash) is not None
                and type(size) is int
                and 0 <= size <= 8 * 1024**3,
                "kache-artifact-meta",
            )
            names.add(name)
            artifact = store / "blobs" / blob_hash[:2] / blob_hash
            # Both versions migrate legacy in-entry artifacts into sharded blobs.
            if not artifact.exists():
                artifact = directory / name
            require(
                artifact.is_file()
                and not artifact.is_symlink()
                and artifact.resolve().is_relative_to(root.resolve())
                and artifact.stat().st_size == size,
                "kache-artifact-unsafe",
            )
        digests[key] = EntrySnapshot(crate_name, len(files))
    return digests


def push_entries(
    binary: str, environment: dict[str, str], root: Path
) -> tuple[int | None, bool]:
    """Push the store's entries, once more for whatever the first pass lost.

    The second pass lists the remote again and pushes only the entries it
    lacks, so a transient connect failure through the mesh costs one retry
    rather than the job. Returns the uploaded count and whether the client
    reported it."""
    uploaded = 0
    reported = True
    for final in (False, True):
        status, stream = attempt(
            [binary, "sync", "--push"], environment, root, "kache-push", 900
        )
        transfer = kache.parse_transfer(stream, "Uploaded")
        if transfer is not None:
            uploaded += transfer.completed
        elif "Nothing to sync." not in stream.splitlines():
            reported = False
        if status == 0 and (transfer is None or transfer.failed == 0):
            return (uploaded if reported else None), reported
        diagnostic("kache-push" if final else "kache-push-retry", stream)
        require(not final, "kache-push")
    raise AssertionError("unreachable")


def publish_kache(
    state: dict[str, Any], root: Path, caller: dict[str, str], evidence: dict[str, Any]
) -> None:
    binary = state["binary"]
    if not binary:
        return
    environment = dict(caller) | state["environment"]
    require(
        kache_version(binary, environment, root) == state["version"],
        "kache-version-changed",
    )
    stop_kache(state, root, caller)
    expected = entry_snapshot(root / "kache")
    evidence["attempted_entries"] = len(expected)
    if not expected:
        return
    original = tomllib.loads(bounded_text(root / "kache.toml"))
    (root / "kache.toml").write_text(
        isolated_config(original, root, state["version"], False), encoding="utf-8"
    )
    environment["KACHE_REMOTE_READONLY"] = "false"
    count, reported = push_entries(binary, environment, root)
    evidence["uploaded_entries"] = count
    evidence["upload_count_reported"] = reported
    # Readback: the client lists the remote again and diffs it against the
    # local store. "Nothing to sync." proves every local entry is listed
    # remotely; a remaining push plan names what the cache did not keep. This
    # transfers no packs: on the mesh a pack readback costs minutes per job.
    plan = command(
        [binary, "sync", "--push", "--dry-run"],
        environment,
        root,
        "kache-readback",
    )
    require(
        "Nothing to sync." in plan.splitlines()
        and not any(line.startswith("Plan: ") for line in plan.splitlines()),
        "kache-readback-incomplete",
    )
    evidence["verified_entries"] = len(expected)


def publish(state_path: str, caller: dict[str, str]) -> dict[str, Any]:
    root = owned_root(state_path, caller)
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "attempted_roots": 0,
        "verified_roots": 0,
        "attempted_entries": 0,
        "verified_entries": 0,
        "uploaded_entries": None,
        "upload_count_reported": False,
        "fresh_nix_uploads": None,
        "cleanup_complete": False,
    }
    token = DEADLINE.set(time.monotonic() + PUBLISH_SECONDS)
    try:
        state = read_state(root, caller)
        evidence["mode"] = state["mode"]
        roots = declared_roots(root)
        evidence["declared_roots"] = len(roots)
        if state["mode"] == "read-write":
            publish_nix(roots, root, caller, evidence)
            publish_kache(state, root, caller, evidence)
        return evidence
    finally:
        DEADLINE.reset(token)
        try:
            cleanup(state_path, caller)
            evidence["cleanup_complete"] = True
        finally:
            # Successful reads prove availability/integrity, not a fresh write.
            # Niks3 supplies no reviewed machine-readable new-upload count.
            print(json.dumps(evidence, sort_keys=True), flush=True)


def terminate(signum: int, frame: object) -> None:
    if CLEANING_UP.get():
        return
    # Repeated runner cancellation must not interrupt the bounded cleanup.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise CacheError("build-cache:terminated")


def main() -> int:
    caller = dict(os.environ)
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        require(
            len(sys.argv) == 2
            and sys.argv[1] in ("restore", "attach", "publish", "cleanup"),
            "operation",
        )
        operation = sys.argv[1]
        if operation == "restore":
            restore(
                caller,
                caller.get("BUILD_CACHE_WORKSPACE", "."),
                caller.get("BUILD_CACHE_MODE", "read"),
                caller.get("BUILD_CACHE_KACHE_BINARY", ""),
            )
        elif operation == "attach":
            attach(
                caller.get("BUILD_CACHE_STATE_PATH", ""),
                caller,
                caller.get("BUILD_CACHE_KACHE_BINARY", ""),
            )
        elif operation == "publish":
            publish(caller.get("BUILD_CACHE_STATE_PATH", ""), caller)
        else:
            cleanup(caller.get("BUILD_CACHE_STATE_PATH", ""), caller)
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        print(
            str(error)
            if isinstance(error, CacheError)
            else "build-cache:invalid-or-unavailable",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
