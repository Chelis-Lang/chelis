"""Real target and consumer checks for portable Chelis cores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from typing import Iterator

from compiler_pipeline_oracle import oracle_environment


ROOT = Path(__file__).resolve().parents[1]
TARGET = "thumbv7em-none-eabi"


class OracleFailure(RuntimeError):
    """A mandatory executable acceptance check failed."""


def _llvm_tool_directory(environment: Mapping[str, str]) -> Path:
    # Query the same compiler selected for the consumer, without a target flag:
    # the LLVM executables run on the host even when they produce Thumb objects.
    command = (environment.get("RUSTC", "rustc"), "--print", "target-libdir")
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=dict(environment), text=True,
            capture_output=True, check=False,
        )
    except OSError as error:
        raise OracleFailure(f"cannot locate compiler LLVM tools: {error}") from error
    if result.returncode != 0:
        raise OracleFailure(
            f"cannot locate compiler LLVM tools (exit {result.returncode}): {result.stderr.strip()}"
        )
    paths = result.stdout.splitlines()
    if len(paths) != 1 or not paths[0]:
        raise OracleFailure("invalid rustc --print target-libdir result")
    libdir = Path(paths[0])
    if not libdir.is_absolute() or libdir.name != "lib" or not libdir.is_dir():
        raise OracleFailure(f"invalid rustc --print target-libdir path: {libdir}")
    directory = libdir.parent / "bin"
    archiver = directory / ("llvm-ar.exe" if os.name == "nt" else "llvm-ar")
    if not archiver.is_file() or not os.access(archiver, os.X_OK):
        raise OracleFailure(
            f"compiler llvm-tools component must provide executable llvm-ar: {archiver}; "
            "install llvm-tools for the selected Rust toolchain"
        )
    return directory


def _environment() -> dict[str, str]:
    environment = oracle_environment()
    with (ROOT / "rust-toolchain.toml").open("rb") as toolchain_file:
        channel = tomllib.load(toolchain_file)["toolchain"]["channel"]
    environment.setdefault("RUSTUP_TOOLCHAIN", channel)
    environment["CARGO_TARGET_DIR"] = os.environ.get(
        "CARGO_TARGET_DIR", str(ROOT / "target" / "agents" / "no-std-cores")
    )
    environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"] = "1"
    environment.setdefault("PYO3_PYTHON", sys.executable)
    tools = _llvm_tool_directory(environment)
    environment["PATH"] = os.pathsep.join([str(tools), environment.get("PATH", "")])
    return environment


def _execute(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    live_stderr: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(map(str, command)), flush=True)
    try:
        result = subprocess.run(
            tuple(map(str, command)),
            cwd=cwd,
            env=_environment() if env is None else dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None if live_stderr else subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise OracleFailure(f"cannot execute {command[0]}: {error}") from error
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result


def run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    required_tests: Iterable[str] = (),
    live_stderr: bool = False,
) -> subprocess.CompletedProcess[str]:
    arguments = tuple(map(str, command))
    test_arguments = None
    if arguments[:3] == ("cargo", "nextest", "run"):
        test_arguments = arguments[3:]
    elif arguments[:2] == ("cargo", "test") and "--doc" not in arguments:
        test_arguments = arguments[2:]
    selected = None
    if test_arguments is not None:
        environment = _environment() if env is None else dict(env)
        environment["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"
        listing_arguments = tuple(
            argument for argument in test_arguments
            if argument not in ("--no-fail-fast", "--fail-fast")
        )
        inventory = run(
            ("cargo", "nextest", "list", *listing_arguments,
             "--run-ignored", "all", "--message-format", "json"),
            cwd=cwd,
            env=environment,
        )
        try:
            selected = _selected_test_names(json.loads(inventory.stdout), required_tests)
        except json.JSONDecodeError as error:
            raise OracleFailure(f"invalid test inventory: {error}") from error
        arguments = ("cargo", "nextest", "run", *test_arguments,
                     "--message-format", "libtest-json-plus")
        env = environment
    result = _execute(arguments, cwd=cwd, env=env, live_stderr=live_stderr)
    if result.returncode != 0:
        raise OracleFailure(f"command failed with exit {result.returncode}: {' '.join(arguments)}")
    if selected is not None:
        _check_test_results(result.stdout, selected)
    return result


def _selected_test_names(metadata: dict, required: Iterable[str] = ()) -> set[str]:
    selected: set[str] = set()
    try:
        for suite in metadata["rust-suites"].values():
            prefix = f"{suite['package-name']}::{suite['binary-name']}$"
            for name, test in suite["testcases"].items():
                status = test["filter-match"]["status"]
                if status == "mismatch":
                    continue
                if status != "matches" or test["kind"] != "test":
                    raise OracleFailure("invalid mandatory test inventory")
                if test["ignored"]:
                    raise OracleFailure(f"ignored mandatory test: {prefix}{name}")
                selected.add(prefix + name)
    except (KeyError, TypeError, AttributeError) as error:
        raise OracleFailure(f"incomplete test inventory: {error}") from error
    if not selected:
        raise OracleFailure("mandatory test inventory is empty")
    for name in required:
        if not any(
            actual == name or actual.endswith("$" + name) or actual.endswith("::" + name)
            for actual in selected
        ):
            raise OracleFailure(f"mandatory test omitted or filtered: {name}")
    return selected


def _check_test_results(stdout: str, selected: set[str]) -> None:
    passed: set[str] = set()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record["type"] != "test" or record["event"] == "started":
                continue
            if record["event"] == "ignored" and record["name"] not in selected:
                continue
            if record["event"] != "ok":
                raise OracleFailure(f"mandatory test did not pass: {record.get('name')}")
            passed.add(record["name"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise OracleFailure(f"invalid structured test evidence: {error}") from error
    if passed != selected:
        missing = sorted(selected - passed)
        unexpected = sorted(passed - selected)
        raise OracleFailure(f"incomplete test evidence: missing={missing}, unexpected={unexpected}")


def _normal_packages(metadata: dict, forbidden: Iterable[str] = ()) -> set[str]:
    try:
        resolution = metadata["resolve"]
        nodes = {node["id"]: node for node in resolution["nodes"]}
        packages = {package["id"]: package for package in metadata["packages"]}
        pending = [resolution["root"]]
        visited: set[str] = set()
        names: set[str] = set()
        std_features: set[str] = set()
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            node = nodes[identity]
            package = packages[identity]
            if "std" in node["features"]:
                std_features.add(package["name"])
            names.add(package["name"])
            for dependency in node["deps"]:
                if not any(kind["kind"] is None for kind in dependency["dep_kinds"]):
                    continue
                target = packages[dependency["pkg"]]
                if any("proc-macro" in item["kind"] for item in target["targets"]):
                    continue
                pending.append(dependency["pkg"])
        rejected = names.intersection(forbidden)
        if rejected:
            raise OracleFailure("forbidden target dependency: " + ", ".join(sorted(rejected)))
        if std_features:
            raise OracleFailure("target std feature: " + ", ".join(sorted(std_features)))
        return names
    except (KeyError, TypeError) as error:
        raise OracleFailure(f"incomplete dependency graph: {error}") from error


@contextmanager
def _package(
    crate: str | Path,
    source: str,
    *,
    executable: bool,
    features: Iterable[str],
    default_features: bool,
    dependencies: str | None = None,
) -> Iterator[Path]:
    candidate = Path(crate)
    path = candidate.resolve() if candidate.is_absolute() else ROOT / "crates" / candidate
    try:
        with (path / "Cargo.toml").open("rb") as manifest_file:
            name = tomllib.load(manifest_file)["package"]["name"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise OracleFailure(f"cannot read core manifest {path}: {error}") from error
    if not isinstance(name, str):
        raise OracleFailure(f"invalid package name in {path}")
    selected = ", ".join(json.dumps(feature) for feature in features)
    dependency = (
        f'{name} = {{ path = {json.dumps(str(path))}, '
        f'default-features = {str(default_features).lower()}, features = [{selected}] }}\n'
    )
    with tempfile.TemporaryDirectory(prefix="chelis-no-std-consumer-") as temporary:
        directory = Path(temporary)
        (directory / "src").mkdir()
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "chelis-no-std-consumer"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n\n[dependencies]\n' + dependency + (dependencies or ""),
            encoding="utf-8",
        )
        (directory / "src" / ("main.rs" if executable else "lib.rs")).write_text(
            source, encoding="utf-8"
        )
        yield directory


def portable(
    crate: str | Path,
    source: str,
    *,
    features: Iterable[str] = (),
    default_features: bool = False,
    forbidden: Iterable[str] = (),
    dependencies: str | None = None,
) -> set[str]:
    with _package(
        crate,
        source,
        executable=False,
        features=features,
        default_features=default_features,
        dependencies=dependencies,
    ) as directory:
        result = run(
            ("cargo", "metadata", "--format-version", "1", "--filter-platform", TARGET),
            cwd=directory,
        )
        try:
            names = _normal_packages(json.loads(result.stdout), forbidden)
        except json.JSONDecodeError as error:
            raise OracleFailure(f"invalid Cargo metadata: {error}") from error
        run(("cargo", "build", "--lib", "--target", TARGET), cwd=directory)
        print(f"portable graph: {', '.join(sorted(names))}", flush=True)
        return names


def consumer(
    crate: str | Path,
    source: str,
    *,
    features: Iterable[str] = (),
    default_features: bool = False,
    dependencies: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    with _package(
        crate,
        source,
        executable=True,
        features=features,
        default_features=default_features,
        dependencies=dependencies,
    ) as directory:
        return run(("cargo", "run", "--quiet"), cwd=directory, env=env)


def _fixture_diagnostics(stdout: str, directory: Path, source: str) -> str:
    fixture = (directory / "src/lib.rs").resolve()
    lines = source.splitlines()
    diagnostics: list[str] = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise OracleFailure(f"invalid negative compile evidence: {error}") from error
        if record.get("reason") != "compiler-message":
            continue
        target = Path(record["target"]["src_path"])
        if not target.is_absolute():
            target = directory / target
        message = record["message"]
        if target.resolve() != fixture or message["level"] != "error":
            continue
        local = False
        for span in message["spans"]:
            index = span["line_start"] - 1
            if (
                span["is_primary"]
                and Path(span["file_name"]).name == "lib.rs"
                and 0 <= index < len(lines)
                and any(text["text"].strip() == lines[index].strip() for text in span["text"])
            ):
                local = True
        if local:
            diagnostics.append(message["message"])
            if message.get("code"):
                diagnostics.append(message["code"]["code"])
            diagnostics.extend(span["label"] for span in message["spans"] if span.get("label"))
            diagnostics.extend(child["message"] for child in message["children"])
    if not diagnostics:
        raise OracleFailure("negative compile fixture has no fixture-local error")
    return "\n".join(diagnostics)


def compile_rejected(
    crate: str | Path,
    source: str,
    *,
    expected: str | Sequence[str],
    features: Iterable[str] = (),
    default_features: bool = False,
    dependencies: str | None = None,
) -> None:
    with _package(
        crate,
        source,
        executable=False,
        features=features,
        default_features=default_features,
        dependencies=dependencies,
    ) as directory:
        result = _execute(("cargo", "check", "--lib", "--message-format", "json"), cwd=directory)
        if result.returncode == 0:
            raise OracleFailure("negative compile fixture unexpectedly passed")
        diagnostic = _fixture_diagnostics(result.stdout, directory, source)
        fragments = (expected,) if isinstance(expected, str) else expected
        if not fragments or any(fragment not in diagnostic for fragment in fragments):
            raise OracleFailure("negative compile fixture failed for an unexpected reason")
