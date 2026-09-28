#!/usr/bin/env python3
"""Execute the runtime bundle acceptance rows and retain byte-bound evidence.

The command deliberately has no row-selection switches. A successful receipt
means every software row ran; accelerator execution is listed separately as
unrun and is never represented as a pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ORACLE_SCHEMA = "chelis-runtime-bundle-oracle/1"
STAGING_RECEIPT_SCHEMA = "chelis-runtime-staging/1"
ARCHIVE_FILE_NAME = "libchelis_runtime.a"
RECEIPT_FILE_NAME = "chelis_runtime.receipt.json"
RUNTIME_DIR_VARIABLE = "CHELIS_RUNTIME_DIR"
LEDGER_PATH_VARIABLE = "CHELIS_OWNERSHIP_LEDGER_PATH"
LEDGER_SCHEMA = "compiled-value-ownership-ledger-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
ORACLE_ROWS = (
    "prerequisites",
    "baseline-build",
    "cli",
    "python",
    "gpu-host-staging",
    "runtime-mutation",
    "dependency-mutation",
    "stale-candidates",
    "instrumented-runtime",
    "distribution",
    "guard",
)
CLI_SOURCE_TEXT = """output = concat(
  [
    to_tensor([1.0f32, 2.0f32]),
    to_tensor([1.0f32, 2.0f32])
  ],
  0i32
)
"""
PYTHON_SOURCE_TEXT = """def join(x: tensor[2, f32]) -> tensor[4, f32] = concat([x, x], 0i32)
"""
EXPECTED_VALUES = [1.0, 2.0, 1.0, 2.0]
RUNTIME_MUTANT_VALUES = [2.0, 1.0, 2.0, 1.0]


class OracleFailure(RuntimeError):
    """A missing prerequisite, failed row, or inconsistent evidence record."""


class CommandResult:
    def __init__(self, argv: Sequence[str], returncode: int, stdout: bytes, stderr: bytes):
        self.argv = list(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class EvidenceRun:
    """Persist every child process and row transition beneath one run directory."""

    def __init__(
        self,
        root: Path,
        tested_head: str | None,
        receipt_path: Path | None = None,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt_path = receipt_path.resolve() if receipt_path else self.root / "receipt.json"
        self.receipt = new_receipt(tested_head, self.root)
        self._next_command = 1
        self.flush()

    def flush(self) -> None:
        atomic_write_json(self.receipt_path, self.receipt)

    def run(
        self,
        label: str,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        expected_returncode: int | None = 0,
    ) -> CommandResult:
        if not argv or any(not isinstance(arg, str) for arg in argv):
            raise OracleFailure(f"{label}: command argv must be a nonempty string list")
        number = self._next_command
        self._next_command += 1
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "command"
        prefix = f"commands/{number:04d}-{safe_label}"
        stdout_path = self.root / f"{prefix}.stdout"
        stderr_path = self.root / f"{prefix}.stderr"
        try:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                env=dict(env) if env is not None else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            launch_error = None
        except OSError as error:
            returncode = None
            stdout = b""
            stderr = str(error).encode("utf-8", errors="replace")
            launch_error = str(error)
        write_bytes_atomically(stdout_path, stdout)
        write_bytes_atomically(stderr_path, stderr)
        expected = "nonzero" if expected_returncode is None else expected_returncode
        status_ok = (
            returncode is not None
            and (returncode != 0 if expected_returncode is None else returncode == expected_returncode)
        )
        record = {
            "label": label,
            "argv": list(argv),
            "cwd": str(cwd.resolve()),
            "expected_returncode": expected,
            "returncode": returncode,
            "status": "passed" if status_ok else "failed",
            "launch_error": launch_error,
            "stdout": artifact_record(self.root, stdout_path),
            "stderr": artifact_record(self.root, stderr_path),
        }
        self.receipt["commands"].append(record)
        if not status_ok:
            self.receipt["overall"] = "failed"
            self.receipt["failure"] = (
                f"{label}: expected exit {expected}, got {returncode}; "
                f"stderr artifact {record['stderr']['path']}"
            )
        self.flush()
        result = CommandResult(argv, returncode if returncode is not None else -1, stdout, stderr)
        if not status_ok:
            raise OracleFailure(self.receipt["failure"])
        return result

    def pass_row(self, name: str, evidence: Mapping[str, Any]) -> None:
        if name not in self.receipt["rows"]:
            raise OracleFailure(f"unknown mandatory oracle row {name!r}")
        if self.receipt["rows"][name]["status"] != "pending":
            raise OracleFailure(f"oracle row {name!r} already has an outcome")
        self.receipt["rows"][name] = {"status": "passed", "evidence": dict(evidence)}
        self.flush()

    def fail_row(self, name: str, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"
        if name in self.receipt["rows"] and self.receipt["rows"][name]["status"] == "pending":
            self.receipt["rows"][name] = {"status": "failed", "reason": message}
        for other, outcome in self.receipt["rows"].items():
            if outcome["status"] == "pending":
                self.receipt["rows"][other] = {
                    "status": "not-run",
                    "reason": f"blocked by failed row {name}",
                }
        self.receipt["overall"] = "failed"
        self.receipt["failure"] = message
        self.flush()

    def finish(self) -> None:
        self.receipt["overall"] = "passed"
        self.receipt.pop("failure", None)
        validate_receipt(self.receipt)
        self.flush()


def new_receipt(tested_head: str | None, artifact_root: Path) -> dict[str, Any]:
    return {
        "schema": ORACLE_SCHEMA,
        "tested_head": tested_head,
        "artifact_root": str(artifact_root.resolve()),
        "overall": "running",
        "prerequisites": {},
        "commands": [],
        "staged_archives": [],
        "native_artifacts": [],
        "rows": {name: {"status": "pending"} for name in ORACLE_ROWS},
        "hardware": hardware_probe_manifest(),
    }


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != ORACLE_SCHEMA:
        raise OracleFailure("oracle receipt has an unknown schema")
    head = receipt.get("tested_head")
    if head is not None and (not isinstance(head, str) or not GIT_SHA_RE.fullmatch(head)):
        raise OracleFailure("oracle receipt tested_head is not a full lowercase Git SHA")
    if receipt.get("overall") not in {"running", "failed", "passed"}:
        raise OracleFailure("oracle receipt has an invalid overall outcome")
    rows = receipt.get("rows")
    if not isinstance(rows, Mapping) or set(rows) != set(ORACLE_ROWS):
        raise OracleFailure("oracle receipt does not contain the exact mandatory rows")
    allowed_statuses = {"pending", "passed", "failed", "not-run"}
    for name, outcome in rows.items():
        if not isinstance(outcome, Mapping) or outcome.get("status") not in allowed_statuses:
            raise OracleFailure(f"oracle row {name!r} has an invalid outcome")
        if outcome.get("status") == "passed" and not isinstance(outcome.get("evidence"), Mapping):
            raise OracleFailure(f"oracle row {name!r} passed without evidence")
    if receipt.get("overall") == "passed":
        if head is None:
            raise OracleFailure("a successful oracle receipt has no tested head")
        missing = [name for name in ORACLE_ROWS if rows[name]["status"] != "passed"]
        if missing:
            raise OracleFailure(f"successful receipt is missing mandatory rows: {missing}")
    artifact_root_value = receipt.get("artifact_root")
    if not isinstance(artifact_root_value, str) or not artifact_root_value:
        raise OracleFailure("oracle receipt has no artifact root")
    artifact_root = Path(artifact_root_value)
    commands = receipt.get("commands")
    if not isinstance(commands, list):
        raise OracleFailure("oracle receipt commands are not an array")
    for record in commands:
        if not isinstance(record, Mapping) or not record.get("argv"):
            raise OracleFailure("oracle receipt contains a command without argv")
        if record.get("status") not in {"passed", "failed"}:
            raise OracleFailure("oracle receipt contains a command without an executed outcome")
        if record.get("returncode") is not None and not isinstance(record.get("returncode"), int):
            raise OracleFailure("oracle receipt command returncode is invalid")
        for stream in ("stdout", "stderr"):
            artifact = record.get(stream)
            if not isinstance(artifact, Mapping) or not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                raise OracleFailure(f"oracle receipt command has no {stream} artifact digest")
            relative = Path(str(artifact.get("path", "")))
            if not artifact.get("path") or relative.is_absolute() or ".." in relative.parts:
                raise OracleFailure(f"oracle receipt command {stream} path is not artifact-root relative")
            artifact_path = artifact_root / relative
            if artifact_path.is_symlink() or not artifact_path.is_file():
                raise OracleFailure(f"oracle receipt command {stream} artifact is missing: {artifact_path}")
            if artifact.get("bytes") != artifact_path.stat().st_size:
                raise OracleFailure(f"oracle receipt command {stream} byte count does not match its artifact")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise OracleFailure(f"oracle receipt command {stream} digest does not match its artifact")
    for collection_name in ("staged_archives", "native_artifacts"):
        collection = receipt.get(collection_name)
        if not isinstance(collection, list):
            raise OracleFailure(f"oracle receipt {collection_name} is not an array")
        for artifact in collection:
            if not isinstance(artifact, Mapping) or not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                raise OracleFailure(f"oracle receipt has a {collection_name} entry without a SHA-256")
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.is_absolute() or artifact_path.is_symlink() or not artifact_path.is_file():
                raise OracleFailure(f"oracle receipt {collection_name} artifact is missing: {artifact_path}")
            if artifact.get("bytes", artifact_path.stat().st_size) != artifact_path.stat().st_size:
                raise OracleFailure(f"oracle receipt {collection_name} byte count does not match its artifact")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise OracleFailure(f"oracle receipt {collection_name} digest does not match its artifact")
            if collection_name == "staged_archives":
                receipt_value = artifact.get("receipt")
                if not isinstance(receipt_value, str) or not receipt_value:
                    raise OracleFailure("oracle staged archive has no receipt path")
                receipt_path = Path(receipt_value)
                if not receipt_path.is_absolute() or receipt_path.is_symlink() or not receipt_path.is_file():
                    raise OracleFailure(f"oracle staged archive receipt is missing: {receipt_path}")
                if artifact.get("receipt_bytes") != receipt_path.stat().st_size:
                    raise OracleFailure(f"oracle staged archive receipt byte count does not match: {receipt_path}")
                receipt_digest = artifact.get("receipt_sha256")
                if not isinstance(receipt_digest, str) or not SHA256_RE.fullmatch(receipt_digest):
                    raise OracleFailure(f"oracle staged archive receipt has no SHA-256: {receipt_path}")
                if sha256_file(receipt_path) != receipt_digest:
                    raise OracleFailure(f"oracle staged archive receipt digest does not match: {receipt_path}")
    hardware = receipt.get("hardware")
    if (
        not isinstance(hardware, list)
        or any(not isinstance(entry, Mapping) for entry in hardware)
        or {entry.get("accelerator") for entry in hardware} != {"HIP", "Metal"}
    ):
        raise OracleFailure("oracle receipt does not bound both accelerator rows")
    for entry in hardware:
        if entry.get("status") != "unrun" or not entry.get("command") or not entry.get("reason"):
            raise OracleFailure("accelerator execution must be explicitly unrun with a command and bound")
    prerequisites = receipt.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise OracleFailure("oracle receipt prerequisites are not an object")
    if receipt.get("overall") == "passed" and prerequisites.get("source_tree_clean") is not True:
        raise OracleFailure("successful oracle receipt does not bind a clean source tree")
    if receipt.get("overall") == "passed" and prerequisites.get("candidate_restored_clean") is not True:
        raise OracleFailure("successful oracle receipt does not bind a restored clean candidate checkout")


def atomic_write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_atomically(path, data)


def write_bytes_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def read_staging_receipt(
    receipt_path: Path,
    archive_path: Path,
    *,
    expected_mode: str | None = None,
    headers_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read staging receipt {receipt_path}: {error}") from error
    if not isinstance(value, dict):
        raise OracleFailure(f"staging receipt {receipt_path} is not a JSON object")
    required = {"schema", "archive", "archive_sha256", "headers", "mode", "chelis_version"}
    missing = required - value.keys()
    if missing:
        raise OracleFailure(f"staging receipt {receipt_path} is missing fields {sorted(missing)}")
    if value["schema"] != STAGING_RECEIPT_SCHEMA:
        raise OracleFailure(f"staging receipt {receipt_path} has schema {value['schema']!r}")
    if not isinstance(value["archive"], str) or Path(value["archive"]).name != value["archive"]:
        raise OracleFailure(f"staging receipt {receipt_path} has an unsafe archive name")
    if archive_path.name != value["archive"]:
        raise OracleFailure(f"staging receipt archive name does not match {archive_path}")
    if not isinstance(value["archive_sha256"], str) or not SHA256_RE.fullmatch(value["archive_sha256"]):
        raise OracleFailure(f"staging receipt {receipt_path} has an invalid archive digest")
    require_archive_digest(archive_path, value["archive_sha256"], label="staged archive")
    if value["mode"] not in {"development", "sealed"}:
        raise OracleFailure(f"staging receipt {receipt_path} has invalid mode {value['mode']!r}")
    if expected_mode is not None and value["mode"] != expected_mode:
        raise OracleFailure(
            f"staging receipt {receipt_path} mode {value['mode']!r} differs from {expected_mode!r}"
        )
    if not isinstance(value["chelis_version"], str) or not value["chelis_version"]:
        raise OracleFailure(f"staging receipt {receipt_path} has no compiler version")
    headers = value["headers"]
    if not isinstance(headers, dict) or not headers:
        raise OracleFailure(f"staging receipt {receipt_path} has no public header digests")
    if headers_dir is not None:
        for name, expected in headers.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise OracleFailure(f"staging receipt {receipt_path} has an unsafe header name")
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                raise OracleFailure(f"staging receipt {receipt_path} has an invalid header digest for {name}")
            header = headers_dir / name
            if header.is_symlink() or not header.is_file():
                raise OracleFailure(f"staged public header is missing or not a regular file: {header}")
            require_archive_digest(header, expected, label=f"staged header {name}")
    return value


def require_archive_digest(path: Path, expected: str, *, label: str) -> str:
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise OracleFailure(f"{label} expected digest is not lowercase SHA-256")
    if path.is_symlink() or not path.is_file():
        raise OracleFailure(f"{label} is missing or not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise OracleFailure(f"{label} digest mismatch at {path}: expected {expected}, got {actual}")
    return actual


def replace_exact(source: bytes, old: bytes, new: bytes, *, path: Path) -> bytes:
    count = source.count(old)
    if count != 1:
        raise OracleFailure(f"{path}: mutation anchor must occur exactly once, found {count}")
    changed = source.replace(old, new, 1)
    if changed == source:
        raise OracleFailure(f"{path}: mutation did not change source bytes")
    return changed


@contextmanager
def mutated_source(
    path: Path, old: bytes, new: bytes, *, additional: Sequence[tuple[bytes, bytes]] = ()
) -> Iterator[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise OracleFailure(f"mutation source is missing or not a regular file: {path}")
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    changed = replace_exact(original, old, new, path=path)
    for other_old, other_new in additional:
        changed = replace_exact(changed, other_old, other_new, path=path)
    record = {
        "path": str(path),
        "original_sha256": sha256_bytes(original),
        "mutated_sha256": sha256_bytes(changed),
        "mode": oct(original_mode),
    }
    try:
        write_bytes_atomically(path, changed)
        path.chmod(original_mode)
        yield record
    finally:
        write_bytes_atomically(path, original)
        path.chmod(original_mode)
        os.utime(path, None)

def require_executable(name: str, *, search_path: str | None = None) -> str:
    found = shutil.which(name, path=search_path)
    if found is None:
        raise OracleFailure(f"required executable {name!r} is not available on PATH")
    return str(Path(found).absolute())


def hardware_probe_manifest() -> list[dict[str, str]]:
    return [
        {
            "accelerator": "HIP",
            "status": "unrun",
            "command": ".venv/bin/python scripts/hip_test.py -p chelis-backend-hip --test integer_abs -- --ignored --test-threads=1",
            "reason": "This oracle proves host staging only; HIP device execution is a separate manual gate and is not claimed here.",
        },
        {
            "accelerator": "Metal",
            "status": "unrun",
            "command": "cargo nextest run -p chelis-backend-metal --test integer_abs_guard --run-ignored all",
            "reason": "This oracle proves host staging only; Metal device execution is a separate manual gate and is not claimed here.",
        },
    ]


def command_env(*, additions: Mapping[str, str] | None = None, path: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop(RUNTIME_DIR_VARIABLE, None)
    env.pop("CHELIS_STYLE_GATE_DISABLE", None)
    env.pop("CARGO_MANIFEST_DIR", None)
    if path is not None:
        env["PATH"] = path
    if additions:
        env.update({name: str(value) for name, value in additions.items()})
    return env


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleFailure(message)


def decode_utf8(data: bytes, *, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OracleFailure(f"{label} is not UTF-8: {error}") from error


def output_json(result: CommandResult, *, label: str) -> dict[str, Any]:
    text = decode_utf8(result.stdout, label=label).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"{label} did not emit one JSON value: {error}; stdout={text!r}") from error
    if not isinstance(value, dict):
        raise OracleFailure(f"{label} did not emit a JSON object")
    return value


def record_staged_archive(
    runner: EvidenceRun,
    *,
    consumer: str,
    target: str,
    archive: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise OracleFailure(f"staging receipt is missing or not a regular file: {receipt_path}")
    record = {
        "consumer": consumer,
        "target": target,
        "path": str(archive.resolve()),
        "receipt": str(receipt_path.absolute()),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
        "sha256": receipt["archive_sha256"],
        "mode": receipt["mode"],
    }
    runner.receipt["staged_archives"].append(record)
    runner.flush()
    return record


def record_native_artifact(
    runner: EvidenceRun,
    *,
    consumer: str,
    kind: str,
    path: Path,
) -> dict[str, Any]:
    record = {
        "consumer": consumer,
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    runner.receipt["native_artifacts"].append(record)
    runner.flush()
    return record


def run_row(runner: EvidenceRun, name: str, action: Any) -> bool:
    try:
        evidence = action()
        if not isinstance(evidence, Mapping):
            raise OracleFailure(f"row {name} returned no evidence object")
        runner.pass_row(name, evidence)
        return True
    except BaseException as error:
        runner.fail_row(name, error)
        return False


def rust_cargo_env(context: Mapping[str, Any]) -> dict[str, str]:
    return command_env(
        additions={
            "CARGO_TARGET_DIR": str(context["cargo_target"]),
            "CARGO_TERM_COLOR": "never",
        }
    )


def build_cli(runner: EvidenceRun, context: dict[str, Any], *, label: str, features: Sequence[str] = ()) -> Path:
    argv = [context["cargo"], "build", "--locked", "-p", "chelis-cli", "--bin", "chelis"]
    if features:
        argv.extend(["--features", ",".join(features)])
    runner.run(label, argv, cwd=context["candidate"], env=rust_cargo_env(context))
    built = context["cargo_target"] / "debug" / ("chelis.exe" if os.name == "nt" else "chelis")
    if not built.is_file():
        raise OracleFailure(f"Cargo reported success without the compiler executable {built}")
    destination = context["bin_dir"] / label
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, destination)
    if os.name != "nt":
        destination.chmod(destination.stat().st_mode | 0o111)
    return destination


def make_python_env(runner: EvidenceRun, context: Mapping[str, Any], label: str) -> dict[str, Path]:
    env_dir = context["run_dir"] / "python-envs" / label
    runner.run(
        f"{label}-venv",
        [context["uv"], "venv", "--python", "3.11", str(env_dir)],
        cwd=context["run_dir"],
        env=command_env(),
    )
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    maturin = env_dir / ("Scripts/maturin.exe" if os.name == "nt" else "bin/maturin")
    runner.run(
        f"{label}-python-dependencies",
        [
            context["uv"],
            "pip",
            "install",
            "--python",
            str(python),
            "maturin",
            "numpy>=2.0",
            "ml_dtypes==0.6.0",
            "packaging>=24",
            "safetensors>=0.5",
        ],
        cwd=context["run_dir"],
        env=command_env(),
    )
    if not python.is_file() or not maturin.is_file():
        raise OracleFailure(f"{label} environment lacks Python or maturin after installation")
    version = runner.run(f"{label}-python-version", [str(python), "--version"], cwd=context["run_dir"])
    if not decode_utf8(version.stdout, label=f"{label} Python version").strip().startswith("Python 3.11."):
        raise OracleFailure(f"{label} environment did not create Python 3.11")
    return {"dir": env_dir, "python": python, "maturin": maturin}


def build_python_extension(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    python_env: Mapping[str, Path],
    *,
    label: str,
    sealed: bool = False,
) -> None:
    argv = [
        str(python_env["maturin"]),
        "develop",
        "--locked",
        "--uv",
    ]
    if sealed:
        argv.extend(["--features", "sealed-runtime"])
    env = rust_cargo_env(context)
    env.pop("CONDA_PREFIX", None)
    env["PYO3_PYTHON"] = str(python_env["python"])
    env["VIRTUAL_ENV"] = str(python_env["dir"])
    runner.run(label, argv, cwd=context["candidate"] / "bindings/python", env=env)


def format_fixtures(runner: EvidenceRun, context: dict[str, Any], cli: Path) -> dict[str, str]:
    source_dir = context["run_dir"] / "inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    cli_source = source_dir / "runtime_bundle_cli.ch"
    python_source = source_dir / "runtime_bundle_python.ch"
    cli_source.write_text(CLI_SOURCE_TEXT, encoding="utf-8")
    python_source.write_text(PYTHON_SOURCE_TEXT, encoding="utf-8")
    for path in (cli_source, python_source):
        runner.run(
            f"format-{path.stem}",
            [str(cli), "fmt", "--inplace", str(path)],
            cwd=context["work_dir"],
            env=command_env(),
        )
    context["cli_source"] = cli_source
    context["python_source"] = python_source
    return {
        "cli_source_sha256": sha256_file(cli_source),
        "python_source_sha256": sha256_file(python_source),
    }


def read_receipt_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read staging receipt {path}: {error}") from error
    if not isinstance(value, dict):
        raise OracleFailure(f"staging receipt {path} is not a JSON object")
    return value


def export_cli_runtime(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    cli: Path,
    *,
    label: str,
    expected_mode: str,
    path: str | None = None,
) -> dict[str, Any]:
    output = context["run_dir"] / "exports" / label
    output.mkdir(parents=True, exist_ok=True)
    runner.run(
        f"{label}-runtime-export",
        [str(cli), "runtime", "export", str(output)],
        cwd=context["work_dir"],
        env=command_env(path=path),
    )
    receipt_path = output / RECEIPT_FILE_NAME
    preliminary = read_receipt_document(receipt_path)
    archive = output / str(preliminary.get("archive", ""))
    receipt = read_staging_receipt(
        receipt_path,
        archive,
        expected_mode=expected_mode,
        headers_dir=output,
    )
    record_staged_archive(
        runner,
        consumer=label,
        target="runtime export",
        archive=archive,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    return {"directory": output, "archive": archive, "receipt_path": receipt_path, "receipt": receipt}


def build_cli_output(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    cli: Path,
    *,
    label: str,
    target: str,
    expected_mode: str = "development",
    expected_digest: str | None = None,
    source: Path | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    source_path = source or context["cli_source"]
    output = context["run_dir"] / "builds" / label
    output.mkdir(parents=True, exist_ok=True)
    result = runner.run(
        f"{label}-build-{target}",
        [str(cli), "build", str(source_path), "--target", target, "--output", str(output)],
        cwd=context["work_dir"],
        env=command_env(path=path),
    )
    receipt_path = output / RECEIPT_FILE_NAME
    preliminary = read_receipt_document(receipt_path)
    archive = output / str(preliminary.get("archive", ""))
    receipt = read_staging_receipt(
        receipt_path,
        archive,
        expected_mode=expected_mode,
        headers_dir=output,
    )
    if result.stderr:
        raise OracleFailure(f"{label} successful CLI build wrote stderr: {result.stderr!r}")
    report = f"Staged runtime {archive} (sha256 {receipt['archive_sha256']})"
    if decode_utf8(result.stdout, label=f"{label} build stdout").splitlines().count(report) != 1:
        raise OracleFailure(f"{label} build did not report its staged archive and digest exactly once")
    if expected_digest is not None and receipt["archive_sha256"] != expected_digest:
        raise OracleFailure(
            f"{label} staged {receipt['archive_sha256']}, expected its carried archive {expected_digest}"
        )
    record_staged_archive(
        runner,
        consumer="chelis CLI",
        target=target,
        archive=archive,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    return {
        "output": output,
        "archive": archive,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def printed_link_argv(build: Mapping[str, Any], *, label: str) -> list[str]:
    text = decode_utf8(build["stdout"], label=f"{label} build stdout")
    compile_lines = [line[len("Compile: ") :] for line in text.splitlines() if line.startswith("Compile: ")]
    if len(compile_lines) != 1:
        raise OracleFailure(f"{label} build did not print exactly one native link command")
    try:
        argv = shlex.split(compile_lines[0])
    except ValueError as error:
        raise OracleFailure(f"{label} native link command is not shell-tokenizable: {error}") from error
    archive = Path(build["archive"]).resolve()
    if sum(Path(arg).resolve() == archive for arg in argv if arg.endswith(archive.name)) != 1:
        raise OracleFailure(f"{label} native link command does not name its one exact staged archive path")
    if any(
        (arg.startswith("-l") and "chelis_runtime" in arg)
        or (arg == "-l" and index + 1 < len(argv) and "chelis_runtime" in argv[index + 1])
        for index, arg in enumerate(argv)
    ):
        raise OracleFailure(f"{label} native link command searches for the runtime instead of linking its staged path")
    return argv


def compile_and_run_cli_output(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    build: Mapping[str, Any],
    *,
    label: str,
    expected_stdout: bytes | None = None,
) -> dict[str, Any]:
    argv = printed_link_argv(build, label=label)
    if "-o" not in argv:
        raise OracleFailure(f"{label} native link command has no output executable")
    out_index = argv.index("-o")
    if out_index + 1 >= len(argv):
        raise OracleFailure(f"{label} native link command has an empty output executable")
    executable = Path(argv[out_index + 1])
    runner.run(f"{label}-native-link", argv, cwd=context["work_dir"], env=command_env())
    if executable.is_symlink() or not executable.is_file():
        raise OracleFailure(f"{label} native linker did not create {executable}")
    run_result = runner.run(
        f"{label}-native-run",
        [str(executable)],
        cwd=context["work_dir"],
        env=command_env(),
    )
    if run_result.stderr:
        raise OracleFailure(f"{label} native program wrote unexpected stderr: {run_result.stderr!r}")
    if expected_stdout is not None and run_result.stdout != expected_stdout:
        raise OracleFailure(
            f"{label} native result differs from the exact witness: "
            f"expected {expected_stdout!r}, got {run_result.stdout!r}"
        )
    native = record_native_artifact(
        runner,
        consumer=label,
        kind="linked executable",
        path=executable,
    )
    return {"argv": argv, "path": str(executable), "sha256": native["sha256"], "stdout": run_result.stdout.decode("utf-8")}


def python_call_script() -> str:
    return '''from pathlib import Path
import json
import re
import sys
import numpy as np
import chelis

source = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])
artifact_dir.mkdir(parents=True, exist_ok=True)
model = chelis.compile_and_load(source, entry_name="join", artifact_dir=artifact_dir, project_root=False)

def observe(candidate):
    output = np.from_dlpack(candidate(np.asarray([1.0, 2.0], dtype=np.float32)))
    return {"dtype": str(output.dtype), "shape": list(output.shape), "values": output.tolist()}

compiled = observe(model)
loaded = chelis.load(model.path)
persisted = observe(loaded)
artifact = Path(model.path)
manifest = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
assert re.fullmatch(r"[0-9a-f]{64}", manifest["runtime_sha256"])
assert re.fullmatch(r"[0-9a-f]{64}", manifest["library_sha256"])
assert __import__("hashlib").sha256(artifact.read_bytes()).hexdigest() == manifest["library_sha256"]
print(json.dumps({"compiled": compiled, "persisted": persisted, "path": str(artifact), "runtime_sha256": manifest["runtime_sha256"], "library_sha256": manifest["library_sha256"]}, sort_keys=True))
'''


def python_rejection_script() -> str:
    return '''from pathlib import Path
import json
import sys
import chelis

mode, operand, *arguments = sys.argv[1:]
artifact_dir = Path(arguments[0]) if arguments else None
try:
    if mode == "compile":
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        options = {"entry_name": "join", "project_root": False}
        if artifact_dir is not None:
            options["artifact_dir"] = artifact_dir
        chelis.compile_and_load(operand, **options)
    elif mode == "load":
        chelis.load(operand)
    else:
        raise AssertionError(f"unknown negative case {mode}")
except chelis.ChelisError as error:
    print(json.dumps({"error": str(error)}, sort_keys=True))
else:
    raise AssertionError(f"{mode} unexpectedly succeeded")
'''


def save_harness(context: Mapping[str, Any], name: str, content: str) -> Path:
    path = context["run_dir"] / "inputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def verify_python_staging(
    artifacts: Path, manifest_runtime_sha256: str, *, expected_mode: str
) -> tuple[Path, Path, dict[str, Any]]:
    receipt_path = artifacts / RECEIPT_FILE_NAME
    archive = artifacts / ARCHIVE_FILE_NAME
    receipt = read_staging_receipt(
        receipt_path, archive, expected_mode=expected_mode, headers_dir=artifacts
    )
    if receipt["archive_sha256"] != manifest_runtime_sha256:
        raise OracleFailure("Python staged archive differs from its persisted runtime digest")
    return archive, receipt_path, receipt


def run_python_call(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    python_env: Mapping[str, Path],
    *,
    label: str,
    source: Path | None = None,
    artifact_dir: Path | None = None,
    path: str | None = None,
    expected_values: Sequence[float] | None = None,
    expected_mode: str = "development",
    additions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_path = source or context["python_source"]
    artifacts = artifact_dir or (context["run_dir"] / "python-artifacts" / label)
    artifacts.mkdir(parents=True, exist_ok=True)
    script = save_harness(context, "python_call_acceptance.py", python_call_script())
    env = command_env(additions=additions, path=path)
    env["PYTHONNOUSERSITE"] = "1"
    result = runner.run(
        f"{label}-python-compile-call-load",
        [str(python_env["python"]), str(script), str(source_path), str(artifacts)],
        cwd=context["work_dir"],
        env=env,
    )
    if result.stderr:
        raise OracleFailure(f"{label} Python witness wrote unexpected stderr: {result.stderr!r}")
    observed = output_json(result, label=f"{label} Python witness")
    values = list(EXPECTED_VALUES if expected_values is None else expected_values)
    expected = {
        "compiled": {"dtype": "float32", "shape": [4], "values": values},
        "persisted": {"dtype": "float32", "shape": [4], "values": values},
    }
    if observed.get("compiled") != expected["compiled"] or observed.get("persisted") != expected["persisted"]:
        raise OracleFailure(f"{label} Python values differ from the exact witness: {observed}")
    artifact = Path(observed.get("path", ""))
    if not artifact.is_file() or artifacts.resolve() not in artifact.resolve().parents:
        raise OracleFailure(f"{label} Python model was not persisted under {artifacts}")
    manifest = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("runtime_sha256") != observed.get("runtime_sha256"):
        raise OracleFailure(f"{label} Python runtime digest differs from its persisted manifest")
    if manifest.get("library_sha256") != sha256_file(artifact):
        raise OracleFailure(f"{label} Python native library digest differs from its persisted manifest")
    archive, receipt_path, receipt = verify_python_staging(
        artifacts, manifest["runtime_sha256"], expected_mode=expected_mode
    )
    record_staged_archive(
        runner,
        consumer=label,
        target="Python compiled artifact",
        archive=archive,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    native = record_native_artifact(
        runner,
        consumer=label,
        kind="Python shared library",
        path=artifact,
    )
    return {
        "values": observed["compiled"]["values"],
        "persisted_values": observed["persisted"]["values"],
        "dtype": observed["compiled"]["dtype"],
        "shape": observed["compiled"]["shape"],
        "runtime_sha256": observed["runtime_sha256"],
        "library_sha256": native["sha256"],
        "artifact": str(artifact.resolve()),
        "artifact_dir": str(artifacts.resolve()),
        "staged_archive": str(archive.resolve()),
        "staging_receipt": str(receipt_path.resolve()),
        "staged_sha256": receipt["archive_sha256"],
        "harness_sha256": sha256_file(script),
    }


def run_python_rejection(
    runner: EvidenceRun,
    context: Mapping[str, Any],
    python_env: Mapping[str, Path],
    *,
    label: str,
    mode: str,
    operand: Path,
    contains: str,
    additions: Mapping[str, str] | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    script = save_harness(context, "python_rejection_acceptance.py", python_rejection_script())
    env = command_env(additions=additions)
    env["PYTHONNOUSERSITE"] = "1"
    result = runner.run(
        label,
        [str(python_env["python"]), str(script), mode, str(operand)]
        + ([str(artifact_dir)] if artifact_dir is not None else []),
        cwd=context["work_dir"],
        env=env,
    )
    if result.stderr:
        raise OracleFailure(f"{label} rejection harness wrote unexpected stderr: {result.stderr!r}")
    observed = output_json(result, label=label)
    message = observed.get("error")
    if not isinstance(message, str) or contains not in message:
        raise OracleFailure(f"{label} did not reject with {contains!r}: {observed}")
    if "load shared library failed" in message:
        raise OracleFailure(f"{label} reached native library open before admission: {message}")
    return {"rejected": True, "diagnostic": message, "harness_sha256": sha256_file(script)}


def compile_c_row(runner: EvidenceRun, context: Mapping[str, Any], cli: Path, label: str, digest: str) -> dict[str, Any]:
    build = build_cli_output(
        runner,
        context,
        cli,
        label=label,
        target="c",
        expected_digest=digest,
    )
    expected = b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n"
    native = compile_and_run_cli_output(
        runner,
        context,
        build,
        label=label,
        expected_stdout=expected,
    )
    eval_result = runner.run(
        f"{label}-eval",
        [str(cli), "eval", "--file", str(context["cli_source"])],
        cwd=context["work_dir"],
        env=command_env(),
    )
    if eval_result.stderr or eval_result.stdout != expected:
        raise OracleFailure(
            f"{label} evaluator exact result differs: stdout={eval_result.stdout!r}, stderr={eval_result.stderr!r}"
        )
    if native["stdout"].encode("utf-8") != eval_result.stdout:
        raise OracleFailure(f"{label} linked C result differs from the evaluator result")
    return {"archive_sha256": digest, "native": native, "exact_stdout": expected.decode("utf-8")}


def cli_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    cli = context["cli_a"]
    exported = export_cli_runtime(runner, context, cli, label="cli-development", expected_mode="development")
    digest = exported["receipt"]["archive_sha256"]
    if context["cli_digest"] != digest:
        raise OracleFailure("CLI build staging differs from its carried runtime export")
    positive = compile_c_row(runner, context, cli, "cli-c-positive", digest)

    override_dir = context["run_dir"] / "negative" / "cli-runtime-override"
    override_dir.mkdir(parents=True, exist_ok=True)
    rejected_output = context["run_dir"] / "builds" / "cli-runtime-override"
    rejected_output.mkdir(parents=True, exist_ok=True)
    result = runner.run(
        "cli-runtime-directory-rejected",
        [str(cli), "build", str(context["cli_source"]), "--target", "c", "--output", str(rejected_output)],
        cwd=context["work_dir"],
        env=command_env(additions={RUNTIME_DIR_VARIABLE: str(override_dir)}),
        expected_returncode=None,
    )
    stderr = decode_utf8(result.stderr, label="CLI runtime-directory rejection stderr")
    if RUNTIME_DIR_VARIABLE not in stderr:
        raise OracleFailure(f"CLI runtime-directory rejection did not name the forbidden variable: {stderr!r}")
    if (rejected_output / RECEIPT_FILE_NAME).exists() or any(
        path.name == ARCHIVE_FILE_NAME for path in rejected_output.iterdir()
    ):
        raise OracleFailure("CLI runtime-directory rejection staged an archive or receipt")
    rejected_export = context["run_dir"] / "negative" / "cli-export-runtime-override"
    rejected_export.mkdir(parents=True, exist_ok=True)
    export_failure = runner.run(
        "cli-runtime-export-directory-rejected",
        [str(cli), "runtime", "export", str(rejected_export)],
        cwd=context["work_dir"],
        env=command_env(additions={RUNTIME_DIR_VARIABLE: str(override_dir)}),
        expected_returncode=None,
    )
    export_stderr = decode_utf8(export_failure.stderr, label="CLI runtime-export rejection stderr")
    if RUNTIME_DIR_VARIABLE not in export_stderr or any(rejected_export.iterdir()):
        raise OracleFailure("CLI runtime export did not refuse the override before staging")

    placeholder = runner.run(
        "bundle-placeholder-stage-test",
        [
            context["cargo"],
            "test",
            "--locked",
            "-p",
            "chelis-runtime-bundle",
            "--lib",
            "a_placeholder_runtime_stages_nothing",
        ],
        cwd=context["candidate"],
        env=rust_cargo_env(context),
    )
    placeholder_text = decode_utf8(placeholder.stdout, label="placeholder stage test output")
    if "a_placeholder_runtime_stages_nothing" not in placeholder_text or "1 passed" not in placeholder_text:
        raise OracleFailure("the placeholder-stage unit test did not execute exactly as required")
    return {
        "carried_sha256": digest,
        "c_build_and_native_run": positive,
        "runtime_directory_negative": {"rejected": True, "diagnostic": stderr.strip()},
        "runtime_export_directory_negative": {"rejected": True, "diagnostic": export_stderr.strip()},
        "placeholder_stage_negative": {"command_recorded": True, "one_test_passed": True},
    }


def python_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    python_a = make_python_env(runner, context, "python-a")
    python_b = make_python_env(runner, context, "python-b")
    context["python_a"] = python_a
    context["python_b"] = python_b
    build_python_extension(runner, context, python_a, label="python-a-development-extension")
    positive = run_python_call(runner, context, python_a, label="python-a-positive")
    # Maturin's link flags can change the runtime archive bytes. Each producer
    # is checked against its own carried, staged, and executed runtime.
    context["python_digest"] = positive["runtime_sha256"]

    override_dir = context["run_dir"] / "negative" / "python-runtime-override"
    override_dir.mkdir(parents=True, exist_ok=True)
    rejected = run_python_rejection(
        runner,
        context,
        python_a,
        label="python-runtime-directory-rejected",
        mode="compile",
        operand=context["python_source"],
        contains=RUNTIME_DIR_VARIABLE,
        additions={RUNTIME_DIR_VARIABLE: str(override_dir)},
    )

    artifact = Path(positive["artifact"])
    artifact_backup = artifact.read_bytes()
    try:
        corrupted = bytearray(artifact_backup)
        if not corrupted:
            raise OracleFailure("persisted Python library is empty and cannot be mutated")
        corrupted[0] ^= 1
        write_bytes_atomically(artifact, bytes(corrupted))
        replaced = run_python_rejection(
            runner,
            context,
            python_a,
            label="python-replaced-library-rejected",
            mode="load",
            operand=artifact,
            contains="SHA-256",
        )
    finally:
        write_bytes_atomically(artifact, artifact_backup)

    missing_digest = {}
    manifest_path = artifact.with_suffix(".json")
    manifest_original = manifest_path.read_bytes()
    manifest = json.loads(manifest_original)
    try:
        for field in ("runtime_sha256", "library_sha256"):
            altered = dict(manifest)
            altered.pop(field, None)
            write_bytes_atomically(manifest_path, (json.dumps(altered) + "\n").encode("utf-8"))
            missing_digest[field] = run_python_rejection(
                runner,
                context,
                python_a,
                label=f"python-missing-{field}-rejected",
                mode="load",
                operand=artifact,
                contains=field,
            )
    finally:
        write_bytes_atomically(manifest_path, manifest_original)

    missing_manifest_library = context["run_dir"] / "negative" / "without-manifest.so"
    shutil.copy2(artifact, missing_manifest_library)
    missing_manifest = run_python_rejection(
        runner,
        context,
        python_a,
        label="python-missing-manifest-rejected",
        mode="load",
        operand=missing_manifest_library,
        contains="read manifest failed",
    )
    return {
        "positive": positive,
        "runtime_directory_negative": rejected,
        "replaced_library_negative": replaced,
        "missing_digest_negatives": missing_digest,
        "missing_manifest_negative": missing_manifest,
    }


def gpu_host_staging_row(runner: EvidenceRun, context: Mapping[str, Any]) -> dict[str, Any]:
    results = {}
    for target, required_files in (
        ("hip", ("chelis_hip_runtime.h", "chelis_device_owner.cpp", "chelis_device_owner.h", "chelis_device_descriptor.h")),
        ("metal", ("chelis_metal_runtime.h",)),
    ):
        build = build_cli_output(
            runner,
            context,
            context["cli_a"],
            label=f"host-stage-{target}",
            target=target,
            expected_digest=context["cli_digest"],
        )
        missing = [name for name in required_files if not (build["output"] / name).is_file()]
        if missing:
            raise OracleFailure(f"{target} build omitted mandatory host staging files: {missing}")
        link_argv = printed_link_argv(build, label=f"host-stage-{target}")
        results[target] = {
            "archive_sha256": build["receipt"]["archive_sha256"],
            "mode": build["receipt"]["mode"],
            "required_files": {name: sha256_file(build["output"] / name) for name in required_files},
            "printed_link_command": link_argv,
            "device_execution": "unrun; see hardware rows",
        }
    return {"host_staging": results, "hardware": hardware_probe_manifest()}


def mutation_build_witness(
    runner: EvidenceRun,
    context: dict[str, Any],
    *,
    label: str,
    changed_path: str,
    old: bytes,
    new: bytes,
    mutant_values: Sequence[float],
    additional_mutations: Sequence[tuple[bytes, bytes]] = (),
) -> dict[str, Any]:
    source_path = context["candidate"] / changed_path
    no_rebuild_dir = context["run_dir"] / "freshness" / label
    no_rebuild_dir.mkdir(parents=True, exist_ok=True)
    python_no_rebuild = context["run_dir"] / "python-artifacts" / f"{label}-stale"
    python_no_rebuild.mkdir(parents=True, exist_ok=True)
    saved_build_artifact = None
    mutation: dict[str, str]
    with mutated_source(source_path, old, new, additional=additional_mutations) as mutation:
        stale_cli = runner.run(
            f"{label}-cli-no-rebuild-freshness",
            [str(context["cli_a"]), "build", str(context["cli_source"]), "--target", "c", "--output", str(no_rebuild_dir)],
            cwd=context["work_dir"],
            env=command_env(),
            expected_returncode=None,
        )
        cli_error = decode_utf8(stale_cli.stderr, label=f"{label} stale CLI diagnostic")
        if changed_path not in cli_error:
            raise OracleFailure(f"stale CLI did not identify changed source {changed_path}: {cli_error!r}")
        if (no_rebuild_dir / RECEIPT_FILE_NAME).exists() or (no_rebuild_dir / ARCHIVE_FILE_NAME).exists():
            raise OracleFailure(f"stale CLI staged files after {changed_path} changed")

        stale_python = run_python_rejection(
            runner,
            context,
            context["python_a"],
            label=f"{label}-python-no-rebuild-freshness",
            mode="compile",
            operand=context["python_source"],
            contains=changed_path,
            artifact_dir=python_no_rebuild,
        )
        if list(python_no_rebuild.iterdir()):
            raise OracleFailure(f"stale Python compile created artifacts after {changed_path} changed")

        cli_b = build_cli(runner, context, label=f"cli-{label}-B")
        export_b = export_cli_runtime(
            runner,
            context,
            cli_b,
            label=f"cli-{label}-B",
            expected_mode="development",
        )
        digest_a = context["cli_digest"]
        digest_b = export_b["receipt"]["archive_sha256"]
        if digest_a == digest_b:
            raise OracleFailure(f"{label} rebuild did not change the carried runtime digest")
        b_build = build_cli_output(
            runner,
            context,
            cli_b,
            label=f"cli-{label}-B-witness",
            target="c",
            expected_digest=digest_b,
        )
        b_native = compile_and_run_cli_output(
            runner,
            context,
            b_build,
            label=f"cli-{label}-B-witness",
            expected_stdout=None,
        )
        expected_mutant_stdout = (
            f"output = tensor(shape=[4], data=[{', '.join(str(value) for value in mutant_values)}])\n"
        ).encode("utf-8")
        observed_mutant_stdout = b_native["stdout"].encode("utf-8")
        if observed_mutant_stdout != expected_mutant_stdout or observed_mutant_stdout == b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n":
            raise OracleFailure(
                f"{label} rebuilt CLI did not produce the exact discriminating wrong result: "
                f"{b_native['stdout']!r}"
            )

        build_python_extension(
            runner,
            context,
            context["python_b"],
            label=f"python-{label}-B-development-extension",
        )
        python_b = run_python_call(
            runner,
            context,
            context["python_b"],
            label=f"python-{label}-B-witness",
            artifact_dir=context["run_dir"] / "python-artifacts" / f"{label}-B",
            expected_values=mutant_values,
        )
        if python_b["runtime_sha256"] == context["python_digest"]:
            raise OracleFailure(f"{label} rebuilt Python extension did not change its carried runtime")
        if python_b["values"] != list(mutant_values):
            raise OracleFailure(
                f"{label} rebuilt Python witness did not produce the exact discriminating result: {python_b['values']}"
            )
        saved_build_artifact = python_b

    cli_a = build_cli(runner, context, label=f"cli-{label}-restored-A")
    context["cli_a"] = cli_a
    export_a = export_cli_runtime(
        runner,
        context,
        cli_a,
        label=f"cli-{label}-restored-A",
        expected_mode="development",
    )
    digest_restored = export_a["receipt"]["archive_sha256"]
    if digest_restored != context["cli_digest"]:
        raise OracleFailure(f"{label} source restoration did not rebuild the original runtime bytes")
    restored_cli = compile_c_row(runner, context, cli_a, f"cli-{label}-restored-A", digest_restored)
    build_python_extension(
        runner,
        context,
        context["python_a"],
        label=f"python-{label}-restored-A-development-extension",
    )
    restored_python = run_python_call(
        runner,
        context,
        context["python_a"],
        label=f"python-{label}-restored-A",
        artifact_dir=context["run_dir"] / "python-artifacts" / f"{label}-restored-A",
    )
    if restored_python["runtime_sha256"] != context["python_digest"]:
        raise OracleFailure(f"{label} restored Python extension did not return to its original runtime")

    cross_runtime = run_python_rejection(
        runner,
        context,
        context["python_a"],
        label=f"python-{label}-cross-runtime-artifact-rejected",
        mode="load",
        operand=Path(saved_build_artifact["artifact"]),
        contains="was linked with runtime",
    )
    return {
        "mutation": mutation,
        "no_rebuild": {"cli": cli_error.strip(), "python": stale_python["diagnostic"]},
        "rebuilt_B": {
            "carried_sha256": digest_b,
            "staged_sha256": b_build["receipt"]["archive_sha256"],
            "archive_path": str(b_build["archive"].resolve()),
            "cli_wrong_stdout": b_native["stdout"],
            "python_wrong_values": python_b["values"],
            "python_carried_sha256": python_b["runtime_sha256"],
            "python_artifact": python_b["artifact"],
            "cross-runtime_persisted_artifact_negative": cross_runtime,
        },
        "restored_A": {
            "carried_sha256": digest_restored,
            "python_carried_sha256": restored_python["runtime_sha256"],
            "cli": restored_cli,
            "python": restored_python,
        },
    }


def runtime_mutation_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    old = b"metadata_or_fail(destination_metadata.byte_offset(destination_index), context);"
    new = b"metadata_or_fail(destination_metadata.byte_offset(destination_index ^ 1), context);"
    # Compiled Python concat lowers to PAD movement plans, not the host
    # chelis_tensor_concat path above. Mutate both executed paths in one B.
    movement_old = b"affine_result(plan.metadata.index(affine_scalar(linear, plan.op)), plan.op)"
    movement_new = b"affine_result(plan.metadata.index(affine_scalar(linear, plan.op) ^ 1), plan.op)"
    return mutation_build_witness(
        runner,
        context,
        label="runtime-mutation",
        changed_path="crates/chelis-runtime/src/lib.rs",
        old=old,
        new=new,
        additional_mutations=((movement_old, movement_new),),
        mutant_values=RUNTIME_MUTANT_VALUES,
    )


def dependency_mutation_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    old = b"usize::try_from(flat).map_err(|_| MetadataError::Overflow(\"tensor index exceeds usize\"))"
    new = b"usize::try_from(flat ^ 1).map_err(|_| MetadataError::Overflow(\"tensor index exceeds usize\"))"
    # CLI concat indexes tensor coordinates through flat_index; compiled
    # Python concat indexes each PAD plan through MovementMetadata::index.
    movement_old = b"target.require_index(flat)?;\n        Ok(flat)"
    movement_new = b"target.require_index(flat)?;\n        Ok(flat ^ 1)"
    return mutation_build_witness(
        runner,
        context,
        label="chelis-abi-flat-index-mutation",
        changed_path="crates/chelis-abi/src/metadata.rs",
        old=old,
        new=new,
        additional_mutations=((movement_old, movement_new),),
        mutant_values=RUNTIME_MUTANT_VALUES,
    )


def stale_candidates_row(runner: EvidenceRun, context: Mapping[str, Any]) -> dict[str, Any]:
    # The retired selectors searched executable-relative deps and, for Python,
    # CARGO_TARGET_DIR/debug/deps. A decoy under debug/build/**/out cannot reach
    # either selector and would make this negative control vacuous.
    good = context["run_dir"] / "exports" / "cli-development" / ARCHIVE_FILE_NAME
    older_archive = Path(context["instrumented_runtime"]["instrumented"]["archive_path"])
    newer_archive = Path(context["runtime_mutation"]["rebuilt_B"]["archive_path"])
    good_digest = context["cli_digest"]
    older_digest = context["instrumented_runtime"]["instrumented"]["archive_sha256"]
    newer_digest = context["runtime_mutation"]["rebuilt_B"]["carried_sha256"]
    require_archive_digest(good, good_digest, label="baseline carried runtime")
    require_archive_digest(older_archive, older_digest, label="instrumented runtime decoy")
    require_archive_digest(newer_archive, newer_digest, label="mutated runtime decoy")
    if len({good_digest, older_digest, newer_digest}) != 3:
        raise OracleFailure("stale-candidate probe requires three distinct runtime configurations")

    roots = {
        "cli_executable_deps": Path(context["cli_a"]).parent / "deps",
        "python_cargo_deps": context["cargo_target"] / "debug" / "deps",
    }
    names = {
        "baseline": ARCHIVE_FILE_NAME,
        "older": f"libchelis_runtime-{older_digest[:16]}-oracle.a",
        "newer": f"libchelis_runtime-{newer_digest[:16]}-oracle.a",
    }
    sources = {"baseline": good, "older": older_archive, "newer": newer_archive}
    installed: list[Path] = []
    cli_deps_created = not roots["cli_executable_deps"].exists()
    planted: dict[str, Any] = {}
    try:
        for consumer, root in roots.items():
            root.mkdir(parents=True, exist_ok=True)
            entries: dict[str, Any] = {}
            for kind, source in sources.items():
                candidate = root / names[kind]
                if candidate.exists() or candidate.is_symlink():
                    raise OracleFailure(f"stale-candidate probe would overwrite an existing archive: {candidate}")
                shutil.copy2(source, candidate)
                installed.append(candidate)
                if kind == "older":
                    os.utime(candidate, ns=(1_000_000_000, 1_000_000_000))
                elif kind == "newer":
                    future = time.time_ns() + 86_400_000_000_000
                    os.utime(candidate, ns=(future, future))
                entries[kind] = {
                    "path": str(candidate),
                    "sha256": sha256_file(candidate),
                    "mtime_ns": candidate.stat().st_mtime_ns,
                }
            if not entries["newer"]["mtime_ns"] > max(
                entry["mtime_ns"] for kind, entry in entries.items() if kind != "newer"
            ):
                raise OracleFailure(f"{consumer} mutated archive is not the newest candidate")
            planted[consumer] = entries
        cli = compile_c_row(
            runner, context, context["cli_a"], "stale-candidates-cli", good_digest
        )
        python = run_python_call(
            runner,
            context,
            context["python_a"],
            label="stale-candidates-python",
            artifact_dir=context["run_dir"] / "python-artifacts" / "stale-candidates",
            additions={"CARGO_TARGET_DIR": str(context["cargo_target"])},
        )
        if python["runtime_sha256"] != context["python_digest"]:
            raise OracleFailure("Python selected a stale build-tree archive instead of its carried runtime")
        return {
            "planted_archives": planted,
            "cli_exact_values": cli["exact_stdout"],
            "python_exact_values": python["values"],
            "carried_sha256": good_digest,
            "python_carried_sha256": context["python_digest"],
            "python_selector_target": str(context["cargo_target"]),
        }
    finally:
        for candidate in installed:
            candidate.unlink()
        if cli_deps_created and roots["cli_executable_deps"].is_dir():
            roots["cli_executable_deps"].rmdir()


def instrumented_runtime_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    baseline = build_cli_output(
        runner,
        context,
        context["cli_a"],
        label="ledger-noninstru-cli",
        target="c",
        expected_digest=context["cli_digest"],
    )
    absent_ledger = context["run_dir"] / "ledger" / "noninstrumented.jsonl"
    absent_ledger.parent.mkdir(parents=True, exist_ok=True)
    native = compile_and_run_cli_output(
        runner,
        context,
        baseline,
        label="ledger-noninstru-native",
        expected_stdout=b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n",
    )
    noninstrumented_run = runner.run(
        "ledger-noninstru-run-with-ledger-path",
        [native["path"]],
        cwd=context["work_dir"],
        env=command_env(additions={LEDGER_PATH_VARIABLE: str(absent_ledger)}),
    )
    expected_stdout = b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n"
    if noninstrumented_run.stdout != expected_stdout or noninstrumented_run.stderr:
        raise OracleFailure(
            f"noninstrumented ledger probe changed native output: "
            f"stdout={noninstrumented_run.stdout!r}, stderr={noninstrumented_run.stderr!r}"
        )
    if absent_ledger.exists():
        raise OracleFailure("noninstrumented runtime wrote an ownership ledger")

    instrumented_cli = build_cli(runner, context, label="cli-ownership-ledger", features=("ownership-ledger",))
    exported = export_cli_runtime(
        runner,
        context,
        instrumented_cli,
        label="cli-ownership-ledger",
        expected_mode="development",
    )
    instrumented_digest = exported["receipt"]["archive_sha256"]
    if instrumented_digest == context["cli_digest"]:
        raise OracleFailure("ownership-ledger feature did not change the carried archive digest")
    built = build_cli_output(
        runner,
        context,
        instrumented_cli,
        label="ledger-instrumented-cli",
        target="c",
        expected_digest=instrumented_digest,
    )
    instrumented_ledger = context["run_dir"] / "ledger" / "instrumented.jsonl"
    instrumented_ledger.parent.mkdir(parents=True, exist_ok=True)
    native_instrumented = compile_and_run_cli_output(
        runner,
        context,
        built,
        label="ledger-instrumented-native",
        expected_stdout=b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n",
    )
    instrumented_run = runner.run(
        "ledger-instrumented-run-with-ledger-path",
        [native_instrumented["path"]],
        cwd=context["work_dir"],
        env=command_env(additions={LEDGER_PATH_VARIABLE: str(instrumented_ledger)}),
    )
    if instrumented_run.stdout != expected_stdout or instrumented_run.stderr:
        raise OracleFailure(
            f"instrumented ledger probe changed native output: "
            f"stdout={instrumented_run.stdout!r}, stderr={instrumented_run.stderr!r}"
        )
    if not instrumented_ledger.is_file():
        raise OracleFailure("ownership-ledger runtime did not write its requested ledger")
    try:
        events = [json.loads(line) for line in instrumented_ledger.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleFailure(f"ownership ledger is not complete JSONL: {error}") from error
    if len(events) < 3 or events[0] != {"event": "header", "schema": LEDGER_SCHEMA}:
        raise OracleFailure(f"ownership ledger has no exact header and allocation events: {events[:2]}")
    summary = events[-1]
    if summary.get("event") != "summary" or summary.get("allocations", 0) <= 0:
        raise OracleFailure(f"ownership ledger has no positive allocation summary: {summary}")
    if summary.get("invalid_operations") != 0:
        raise OracleFailure(f"ownership ledger reports invalid transitions: {summary}")
    if not any(event.get("event") == "allocate" for event in events[1:-1]):
        raise OracleFailure("ownership ledger has a summary but no recorded allocation transition")
    return {
        "noninstrumented": {"ledger_exists": False, "native": native},
        "instrumented": {
            "archive_sha256": instrumented_digest,
            "archive_path": str(exported["archive"].resolve()),
            "ledger_sha256": sha256_file(instrumented_ledger),
            "event_count": len(events),
            "summary": summary,
            "native": native_instrumented,
        },
    }


def nix_output(runner: EvidenceRun, context: Mapping[str, Any], installable: str) -> Path:
    result = runner.run(
        f"nix-build-{installable.replace('.', '-').replace('#', '-')}",
        [context["nix"], "build", "--no-link", "--print-out-paths", installable],
        cwd=context["candidate"],
        env=command_env(),
    )
    lines = [line.strip() for line in decode_utf8(result.stdout, label=f"Nix output {installable}").splitlines() if line.strip()]
    paths = [line for line in lines if line.startswith("/")]
    if len(paths) != 1:
        raise OracleFailure(f"Nix build {installable} returned {len(paths)} store paths: {lines}")
    path = Path(paths[0])
    if not path.is_dir():
        raise OracleFailure(f"Nix output {installable} is not a directory: {path}")
    return path


def python_wheel_build(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    wheel_env = make_python_env(runner, context, "python-wheel")
    wheel_dir = context["run_dir"] / "wheel"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    env = rust_cargo_env(context)
    env["PYO3_PYTHON"] = str(wheel_env["python"])
    result = runner.run(
        "maturin-build-sealed-wheel",
        [
            str(wheel_env["maturin"]),
            "build",
            "--locked",
            "--features",
            "extension-module,sealed-runtime",
            "--interpreter",
            str(wheel_env["python"]),
            "--out",
            str(wheel_dir),
        ],
        cwd=context["candidate"] / "bindings/python",
        env=env,
    )
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise OracleFailure(f"local Maturin build produced {len(wheels)} wheels: {wheels}")
    wheel = wheels[0]
    runner.run(
        "install-local-wheel",
        [context["uv"], "pip", "install", "--python", str(wheel_env["python"]), "--no-deps", str(wheel)],
        cwd=context["work_dir"],
        env=command_env(),
    )
    return {"env": wheel_env, "wheel": wheel, "wheel_sha256": sha256_file(wheel), "build_stdout": result.stdout.decode("utf-8", errors="replace")}


@contextmanager
def checkout_unavailable(runner: EvidenceRun, context: Mapping[str, Any]) -> Iterator[Path]:
    candidate = Path(context["candidate"])
    hidden = Path(context["run_dir"]) / "relocated-away-checkout"
    if hidden.exists():
        raise OracleFailure(f"checkout-removal probe path already exists: {hidden}")
    before = runner.run(
        "verify-candidate-clean-before-source-free-probe",
        [context["git"], "status", "--porcelain", "--untracked-files=all"],
        cwd=candidate,
    )
    status = decode_utf8(before.stdout, label="candidate status before source-free probe")
    if status:
        raise OracleFailure(f"candidate checkout is dirty before source-free probe: {status}")
    os.replace(candidate, hidden)
    try:
        if candidate.exists():
            raise OracleFailure("candidate checkout path still exists during source-free probe")
        yield hidden
    finally:
        os.replace(hidden, candidate)


def restricted_product_path(context: Mapping[str, Any]) -> str:
    compiler = context["cc"]
    directories = [str(Path(compiler).parent), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    unique = list(dict.fromkeys(directory for directory in directories if Path(directory).is_dir()))
    path = os.pathsep.join(unique)
    if shutil.which("chelis", path=path) is not None:
        raise OracleFailure("restricted source-free PATH unexpectedly exposes a neighboring compiler CLI")
    return path


def verify_package_runtime(package: Path, export_receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    archive = package / "lib" / ARCHIVE_FILE_NAME
    digest = require_archive_digest(archive, export_receipt["archive_sha256"], label=f"{label} archive")
    headers = {}
    for name, expected in export_receipt["headers"].items():
        header = package / "include" / name
        headers[name] = require_archive_digest(header, expected, label=f"{label} header {name}")
    return {"archive": str(archive), "archive_sha256": digest, "headers": headers}


def cross_package_runtime(package: Path, destination: Path, replacement: Path) -> Path:
    shutil.copytree(package, destination, symlinks=True)
    copied_lib = destination / "lib"
    if copied_lib.is_symlink() or not copied_lib.is_dir():
        raise OracleFailure(f"crossed package has no independent lib directory: {copied_lib}")
    copied_lib.chmod(stat.S_IMODE(copied_lib.stat().st_mode) | stat.S_IWUSR)
    crossed = copied_lib / ARCHIVE_FILE_NAME
    write_bytes_atomically(crossed, replacement.read_bytes())
    return crossed


def distribution_row(runner: EvidenceRun, context: dict[str, Any]) -> dict[str, Any]:
    cli_package = nix_output(runner, context, ".#chelis")
    runtime_package = nix_output(runner, context, ".#chelis-runtime")
    package_cli = cli_package / "bin" / ("chelis.exe" if os.name == "nt" else "chelis")
    if package_cli.is_symlink():
        raise OracleFailure(f"Nix compiler package CLI is a symlink: {package_cli}")
    if not package_cli.is_file() or not os.access(package_cli, os.X_OK):
        raise OracleFailure(f"Nix compiler package lacks an executable CLI: {package_cli}")
    if not (runtime_package / "lib" / ARCHIVE_FILE_NAME).is_file():
        raise OracleFailure(f"Nix runtime package lacks its archive: {runtime_package}")

    nix_export = export_cli_runtime(
        runner,
        context,
        package_cli,
        label="nix-sealed-cli",
        expected_mode="sealed",
    )
    sealed_digest = nix_export["receipt"]["archive_sha256"]
    compiler_package_files = verify_package_runtime(cli_package, nix_export["receipt"], label="Nix compiler package")
    runtime_package_files = verify_package_runtime(runtime_package, nix_export["receipt"], label="Nix runtime package")

    crossed_dir = context["run_dir"] / "distribution" / "crossed-runtime-package"
    mutation_archive = Path(context["runtime_mutation"]["rebuilt_B"]["archive_path"])
    cross_package_runtime(runtime_package, crossed_dir, mutation_archive)
    try:
        verify_package_runtime(crossed_dir, nix_export["receipt"], label="crossed Nix runtime package")
    except OracleFailure as error:
        crossed_rejected = str(error)
        if "digest mismatch" not in crossed_rejected:
            raise OracleFailure(f"crossed package rejected for a reason other than its archive bytes: {crossed_rejected}") from error
    else:
        raise OracleFailure("crossed archive was admitted by the package check")

    relocated = context["run_dir"] / "distribution" / "relocated-cli"
    shutil.copytree(cli_package, relocated, symlinks=True)
    relocated_cli = relocated / "bin" / package_cli.name
    if relocated_cli.is_symlink() or not relocated_cli.is_file():
        raise OracleFailure("relocated sealed CLI is not a self-contained regular executable")

    wheel = python_wheel_build(runner, context)
    path_without_cli = restricted_product_path(context)
    with checkout_unavailable(runner, context) as moved_checkout:
        relocated_export = export_cli_runtime(
            runner,
            context,
            relocated_cli,
            label="relocated-sealed-cli",
            expected_mode="sealed",
            path=path_without_cli,
        )
        if relocated_export["receipt"]["archive_sha256"] != sealed_digest:
            raise OracleFailure("relocated CLI carries different bytes from the Nix sealed package export")
        relocated_build = build_cli_output(
            runner,
            context,
            relocated_cli,
            label="relocated-sealed-cli-build",
            target="c",
            expected_mode="sealed",
            expected_digest=sealed_digest,
            path=path_without_cli,
        )
        relocated_native = compile_and_run_cli_output(
            runner,
            context,
            relocated_build,
            label="relocated-sealed-cli-native",
            expected_stdout=b"output = tensor(shape=[4], data=[1.0, 2.0, 1.0, 2.0])\n",
        )
        location_env = command_env(path=path_without_cli)
        location_env["PYTHONNOUSERSITE"] = "1"
        wheel_artifacts = context["run_dir"] / "python-artifacts" / "source-free-wheel"
        wheel_result = run_python_call(
            runner,
            context,
            wheel["env"],
            label="source-free-wheel",
            artifact_dir=wheel_artifacts,
            path=path_without_cli,
            expected_mode="sealed",
        )
        package_probe = save_harness(
            context,
            "wheel-location.py",
            "import json, pathlib, chelis, chelis._native\n"
            "print(json.dumps({'package': str(pathlib.Path(chelis.__file__).resolve()), 'extension': str(pathlib.Path(chelis._native.__file__).resolve())}, sort_keys=True))\n",
        )
        location_result = runner.run(
            "source-free-wheel-location",
            [str(wheel["env"]["python"]), str(package_probe)],
            cwd=context["work_dir"],
            env=location_env,
        )
        locations = output_json(location_result, label="source-free wheel import location")
        env_root = wheel["env"]["dir"].resolve()
        for key in ("package", "extension"):
            imported = Path(locations[key]).resolve()
            if env_root not in imported.parents:
                raise OracleFailure(f"source-free wheel imported {key} from outside its venv: {imported}")
        if context["candidate"].exists():
            raise OracleFailure("candidate checkout was available during source-free distribution tests")

    return {
        "nix": {
            "chelis_package": str(cli_package),
            "chelis_runtime_package": str(runtime_package),
            "export_sha256": sealed_digest,
            "compiler_package_archive_sha256": compiler_package_files["archive_sha256"],
            "compiler_package_headers": compiler_package_files["headers"],
            "runtime_package_headers": runtime_package_files["headers"],
            "package_archive_sha256": runtime_package_files["archive_sha256"],
            "relocated_cli_native": relocated_native,
            "relocated_cli_mode": relocated_export["receipt"]["mode"],
            "relocated_checkout_path_absent": True,
            "crossed_archive_rejected": crossed_rejected,
        },
        "wheel": {
            "path": str(wheel["wheel"]),
            "sha256": wheel["wheel_sha256"],
            "python": wheel_result,
            "package_location": locations,
            "checkout_path_absent": True,
            "wheel_build_stdout": wheel["build_stdout"],
            "runtime_sha256": wheel_result["runtime_sha256"],
            "restricted_path": path_without_cli,
            "neighboring_cli_visible": shutil.which("chelis", path=path_without_cli) is not None,
        },
        "relocated_checkout_path": str(moved_checkout),
    }

def require_no_reviewed_lookups(rows: Sequence[Any]) -> None:
    survivors = [f"{row.path} [{row.pattern}]" for row in rows if row.disposition == "lookup"]
    if survivors:
        raise OracleFailure("runtime archive guard still reviews active lookups:\n" + "\n".join(survivors))


def guard_row(runner: EvidenceRun, context: Mapping[str, Any]) -> dict[str, Any]:
    guard_root = Path(context["candidate"])
    sys.path.insert(0, str(guard_root / "scripts"))
    try:
        import check_runtime_archive_lookups as runtime_guard
    except ImportError as error:
        raise OracleFailure(f"cannot import the runtime archive guard: {error}") from error
    clean = runtime_guard.check(guard_root)
    if clean:
        raise OracleFailure("runtime archive guard is not clean:\n" + "\n".join(clean))
    require_no_reviewed_lookups(runtime_guard.REVIEWED)
    probe_name = f"runtime-bundle-oracle-guard-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix=probe_name, dir=guard_root) as directory:
        probe = Path(directory) / "lookup.rs"
        probe.write_text(
            f'let candidate = build_dir.join("{ARCHIVE_FILE_NAME}");\n',
            encoding="utf-8",
        )
        planted = runtime_guard.check(guard_root)
        relative = str(probe.relative_to(guard_root))
        if not any(relative in error and "unreviewed line" in error for error in planted):
            raise OracleFailure(f"runtime archive guard admitted the planted lookup: {planted}")
    after = runtime_guard.check(guard_root)
    if after:
        raise OracleFailure("runtime archive guard did not return clean after probe restoration:\n" + "\n".join(after))
    return {
        "clean_scan": "passed",
        "planted_lookup": "rejected",
        "probe_removed": True,
        "current_tree_status": "candidate clone clean after probe",
    }


def source_tree_status(runner: EvidenceRun, root: Path) -> str:
    result = runner.run(
        "source-tree-status",
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
    )
    return decode_utf8(result.stdout, label="source tree status")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    target_root = ROOT / "target" / "runtime-bundle"
    target_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = target_root / "runs" / run_id
    runner = EvidenceRun(run_dir, None, target_root / "receipt.json")
    try:
        head_result = runner.run("tested-head", ["git", "rev-parse", "HEAD"], cwd=ROOT)
        head = decode_utf8(head_result.stdout, label="tested Git head").strip()
        if not GIT_SHA_RE.fullmatch(head):
            raise OracleFailure(f"Git returned a non-full tested head: {head!r}")
        runner.receipt["tested_head"] = head
        runner.flush()

        context: dict[str, Any] = {
            "root": ROOT,
            "run_dir": run_dir,
            "target_root": target_root,
            "cargo_target": run_dir / "cargo-target",
            "bin_dir": run_dir / "bin",
            "work_dir": run_dir / "working-directory",
        }
        context["bin_dir"].mkdir(parents=True, exist_ok=True)
        context["work_dir"].mkdir(parents=True, exist_ok=True)

        def prerequisites() -> Mapping[str, Any]:
            if sys.version_info[:2] != (3, 11):
                raise OracleFailure(f"oracle requires this checkout's Python 3.11; got {sys.version}")
            resolved = {}
            for name in ("git", "cargo", "uv", "nix", "cc"):
                resolved[name] = require_executable(name)
            context.update(resolved)
            context["cargo"] = resolved["cargo"]
            context["uv"] = resolved["uv"]
            context["nix"] = resolved["nix"]
            context["cc"] = resolved["cc"]
            runner.receipt["prerequisites"].update(
                {
                    "python": sys.version,
                    "python_executable": str(Path(sys.executable).resolve()),
                    "executables": resolved,
                }
            )
            status = source_tree_status(runner, ROOT)
            if status:
                runner.receipt["prerequisites"]["source_tree_clean"] = False
                runner.flush()
                raise OracleFailure(f"run the oracle from a clean committed head; source tree is dirty:\n{status}")
            runner.receipt["prerequisites"]["source_tree_clean"] = True
            clone = run_dir / "candidate-checkout"
            runner.run(
                "clone-candidate-head",
                [resolved["git"], "clone", "--shared", "--no-checkout", str(ROOT), str(clone)],
                cwd=ROOT,
            )
            runner.run(
                "checkout-candidate-head",
                [resolved["git"], "-C", str(clone), "checkout", "--detach", head],
                cwd=ROOT,
            )
            candidate_head = runner.run(
                "verify-candidate-head",
                [resolved["git"], "-C", str(clone), "rev-parse", "HEAD"],
                cwd=ROOT,
            )
            if decode_utf8(candidate_head.stdout, label="candidate Git head").strip() != head:
                raise OracleFailure("candidate build checkout does not match the tested head")
            candidate_status = source_tree_status(runner, clone)
            if candidate_status:
                raise OracleFailure(f"candidate checkout is not clean:\n{candidate_status}")
            context["candidate"] = clone
            context["mutation_candidate"] = clone
            return {
                "source_tree_clean": True,
                "tested_head": head,
                "candidate_checkout": str(clone),
                "candidate_head": head,
                "executables": resolved,
            }

        if not run_row(runner, "prerequisites", prerequisites):
            print(f"RUNTIME BUNDLE ORACLE: FAIL ({runner.receipt_path})", file=sys.stderr)
            return 1

        def baseline_build() -> Mapping[str, Any]:
            cli = build_cli(runner, context, label="cli-default-A")
            context["cli_a"] = cli
            exported = export_cli_runtime(runner, context, cli, label="cli-development", expected_mode="development")
            context["cli_digest"] = exported["receipt"]["archive_sha256"]
            fixtures = format_fixtures(runner, context, cli)
            return {
                "cli": str(cli),
                "carried_sha256": context["cli_digest"],
                "fixture_hashes": fixtures,
            }

        if not run_row(runner, "baseline-build", baseline_build):
            print(f"RUNTIME BUNDLE ORACLE: FAIL ({runner.receipt_path})", file=sys.stderr)
            return 1

        def runtime_mutation() -> Mapping[str, Any]:
            evidence = runtime_mutation_row(runner, context)
            context["runtime_mutation"] = evidence
            return evidence

        def dependency_mutation() -> Mapping[str, Any]:
            evidence = dependency_mutation_row(runner, context)
            context["dependency_mutation"] = evidence
            return evidence

        def instrumented_runtime() -> Mapping[str, Any]:
            evidence = instrumented_runtime_row(runner, context)
            context["instrumented_runtime"] = evidence
            return evidence
        rows = (
            ("cli", lambda: cli_row(runner, context)),
            ("python", lambda: python_row(runner, context)),
            ("gpu-host-staging", lambda: gpu_host_staging_row(runner, context)),
            ("runtime-mutation", runtime_mutation),
            ("dependency-mutation", dependency_mutation),
            ("instrumented-runtime", instrumented_runtime),
            ("stale-candidates", lambda: stale_candidates_row(runner, context)),
            ("distribution", lambda: distribution_row(runner, context)),
            ("guard", lambda: guard_row(runner, context)),
        )
        for name, action in rows:
            if not run_row(runner, name, action):
                print(f"RUNTIME BUNDLE ORACLE: FAIL ({runner.receipt_path})", file=sys.stderr)
                return 1

        final_status = source_tree_status(runner, ROOT)
        if final_status:
            raise OracleFailure(f"oracle left the source tree dirty:\n{final_status}")
        candidate_status = source_tree_status(runner, context["candidate"])
        if candidate_status:
            raise OracleFailure(f"oracle did not restore the mutation checkout:\n{candidate_status}")
        runner.receipt["prerequisites"]["candidate_restored_clean"] = True
        runner.finish()
    except BaseException as error:
        current = next(
            (name for name, row in runner.receipt["rows"].items() if row["status"] == "pending"),
            None,
        )
        if current is not None:
            runner.fail_row(current, error)
        else:
            runner.receipt["overall"] = "failed"
            runner.receipt["failure"] = f"{type(error).__name__}: {error}"
            runner.flush()
        print(f"RUNTIME BUNDLE ORACLE: FAIL ({runner.receipt_path})\n{error}", file=sys.stderr)
        return 1
    print(f"RUNTIME BUNDLE ORACLE: PASS ({runner.receipt_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
