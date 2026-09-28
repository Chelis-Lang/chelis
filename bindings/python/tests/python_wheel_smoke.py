"""Prove editable freshness and sealed-wheel execution outside disposable sources.

The smoke keeps its editable source copy at a stable task-owned path outside
Cargo's target directory, refreshes it for each run, and removes it before
returning. This bounded smoke is not the aggregate runtime-artifact oracle.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSUMER_SCRIPT = Path(__file__).with_name("python_wheel_consumer.py")
RUNTIME_DIR_ENV = "CHELIS_RUNTIME_DIR"
RUNTIME_SOURCE = Path("crates/chelis-runtime/src/lib.rs")
CROSS_BUILD_MARKER = b"bits: u64::from_ne_bytes(seed.to_ne_bytes()),"
CROSS_BUILD_REPLACEMENT = b"bits: u64::from_ne_bytes(seed.to_ne_bytes()).wrapping_add(1),"


class SmokeError(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    logs_dir: Path,
    label: str,
) -> subprocess.CompletedProcess[str]:
    print(f"wheel smoke: {label} starting", file=sys.stderr, flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = logs_dir / f"{label}.json"
    write_json(
        log_path,
        {
            "command": command,
            "cwd": str(cwd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    print(
        f"wheel smoke: {label} exited {result.returncode}; log {log_path}",
        file=sys.stderr,
        flush=True,
    )
    if result.returncode:
        raise SmokeError(
            f"{shlex.join(command)} failed with exit {result.returncode}; "
            f"command output: {log_path}"
        )
    return result


def compiler_path() -> Path:
    configured = os.environ.get("CHELIS_CC")
    if configured:
        resolved = shutil.which(configured) if not Path(configured).is_absolute() else configured
        if resolved is None:
            raise SmokeError(f"CHELIS_CC is not executable: {configured}")
        return Path(resolved).resolve()
    candidates = ("clang", "cc") if sys.platform == "darwin" else ("gcc", "cc")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved).resolve()
    raise SmokeError(f"no native C compiler found; tried {', '.join(candidates)}")


def source_copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        ".devenv",
        "target",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    current = Path(directory).resolve()
    if current in {REPO_ROOT / "bindings/python", REPO_ROOT / "py"}:
        ignored.update({"build", "dist"})
    result = {
        name
        for name in names
        if name in ignored or name.endswith((".pyc", ".pyo", ".egg-info", ".dist-info"))
    }
    result.update(name for name in names if name.endswith((".so", ".dylib", ".pyd")))
    return result


def validate_source_links(root: Path) -> None:
    root = root.resolve()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise SmokeError(f"source copy has an unresolved symlink: {path}") from error
            if not target.is_relative_to(root):
                raise SmokeError(
                    f"source copy symlink escapes its disposable root: {path} -> {target}"
                )


def copy_source_tree(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        raise SmokeError(f"source copy destination is a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise SmokeError(
                f"source copy destination is not a directory: {destination}"
            )
        shutil.rmtree(destination)
    try:
        shutil.copytree(
            REPO_ROOT, destination, ignore=source_copy_ignore, symlinks=True
        )
        validate_source_links(destination)
    except BaseException:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise


def editable_source_directory(target: Path, scratch: Path) -> Path:
    source = scratch / "editable-source"
    if source.is_symlink():
        raise SmokeError(f"editable source copy path is a symlink: {source}")
    resolved_source = source.resolve()
    target = target.resolve()
    if resolved_source.is_relative_to(target) or target.is_relative_to(resolved_source):
        raise SmokeError(
            f"CARGO_TARGET_DIR must be separate from the editable source copy: {target}"
        )
    return source


def refresh_editable_runtime_bundle(
    *,
    target: Path,
    source_root: Path,
    env: dict[str, str],
    logs_dir: Path,
    label: str,
) -> None:
    # The runtime build record and the bundle embed source bytes and path.
    # A fresh copy at the same scratch path can be older than Cargo's cached
    # fingerprints, so rebuild their crates and the extension on every run.
    run(
        [
            "cargo",
            "clean",
            "--release",
            "--manifest-path",
            str(source_root / "Cargo.toml"),
            "--package",
            "chelis-runtime",
            "--package",
            "chelis-runtime-bundle",
            "--package",
            "chelis-python",
        ],
        cwd=source_root,
        env=env,
        logs_dir=logs_dir,
        label=label,
    )


def checkout_free_entries(value: str, checkout_root: Path, consumer_cwd: Path) -> list[str]:
    checkout_root = checkout_root.resolve()
    consumer_cwd = consumer_cwd.resolve()
    entries = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        resolved = path.resolve() if path.is_absolute() else (consumer_cwd / path).resolve()
        if not resolved.is_relative_to(checkout_root):
            entries.append(str(resolved))
    return entries


def checkout_free_path(checkout_root: Path, compiler: Path, consumer_cwd: Path) -> str:
    entries = [
        entry
        for entry in checkout_free_entries(
            os.environ.get("PATH", ""), checkout_root, consumer_cwd
        )
        if shutil.which("chelis", path=entry) is None
    ]
    compiler_dir = str(compiler.parent)
    if compiler_dir not in entries and shutil.which("chelis", path=compiler_dir) is None:
        entries.append(compiler_dir)
    return os.pathsep.join(entries)


def target_directory() -> Path:
    configured = os.environ.get("CARGO_TARGET_DIR")
    target = (
        Path(configured).expanduser()
        if configured
        else REPO_ROOT / "target" / "python-wheel-smoke"
    )
    if not target.is_absolute():
        raise SmokeError("CARGO_TARGET_DIR must be absolute for the wheel smoke")
    target = target.resolve()
    task_target_root = (REPO_ROOT / "target" / "python-wheel-smoke").resolve()
    if target not in (task_target_root, task_target_root / "cargo"):
        raise SmokeError(
            f"CARGO_TARGET_DIR must be the dedicated Python wheel smoke target: {target}"
        )
    return target


def common_repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SmokeError(
            f"cannot determine shared repository root: {result.stderr.strip()}"
        )
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = REPO_ROOT / common_dir
    return common_dir.resolve().parent


def build_environment(target: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYO3_PYTHON"] = sys.executable
    env["UV_CACHE_DIR"] = str(target / "uv-cache")
    env.setdefault("CARGO_PROFILE_RELEASE_DEBUG", "0")
    env.setdefault("CARGO_INCREMENTAL", "0")
    return env


def consumer_environment(
    *, checkout_root: Path, compiler: Path, cwd: Path, scratch: Path
) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        RUNTIME_DIR_ENV,
        "CHELIS_REEF_HOME",
        "CARGO_TARGET_DIR",
        "UV_CACHE_DIR",
        "CARGO_HOME",
        "CARGO_MANIFEST_DIR",
        "PYO3_PYTHON",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ):
        env.pop(key, None)
    env["PATH"] = checkout_free_path(checkout_root, compiler, cwd)
    if shutil.which("chelis", path=env["PATH"]) is not None:
        raise SmokeError("the installed Python consumer PATH still exposes a chelis executable")
    for key in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
    ):
        value = env.get(key)
        if value is None:
            continue
        entries = checkout_free_entries(value, checkout_root, cwd)
        if entries:
            env[key] = os.pathsep.join(entries)
        else:
            env.pop(key)
    env["HOME"] = str(scratch / "home")
    env["TMPDIR"] = str(scratch / "tmp")
    env["CHELIS_CC"] = str(compiler)
    return env


def consumer_command(
    *,
    python: Path,
    driver: Path,
    mode: str,
    work_root: Path,
    checkout_root: Path,
    result_path: Path,
    source_root: Path | None = None,
    source_copy: Path | None = None,
    compile_result: Path | None = None,
    library: Path | None = None,
    artifact_dir: Path | None = None,
    expected_runtime_sha256: str | None = None,
    developer_target: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        "-I",
        str(driver),
        "--mode",
        mode,
        "--work-root",
        str(work_root),
        "--checkout-root",
        str(checkout_root),
        "--runtime-dir-variable",
        RUNTIME_DIR_ENV,
        "--result",
        str(result_path),
    ]
    if source_root is not None:
        command.extend(("--source-root", str(source_root)))
    if source_copy is not None:
        command.extend(("--source-copy", str(source_copy)))
    if compile_result is not None:
        command.extend(("--compile-result", str(compile_result)))
    if library is not None:
        command.extend(("--library", str(library)))
    if artifact_dir is not None:
        command.extend(("--artifact-dir", str(artifact_dir)))
    if expected_runtime_sha256 is not None:
        command.extend(("--expected-runtime-sha256", expected_runtime_sha256))
    if developer_target is not None:
        command.extend(("--developer-target", str(developer_target)))
    return command


def make_venv(
    uv: str, path: Path, *, env: dict[str, str], scratch: Path, logs_dir: Path, label: str
) -> Path:
    run(
        [uv, "venv", "--python", sys.executable, str(path)],
        cwd=scratch,
        env=env,
        logs_dir=logs_dir,
        label=f"{label}-venv",
    )
    python = path / "bin" / "python"
    if not python.is_file():
        raise SmokeError(f"uv did not create the expected Python interpreter: {python}")
    return python


def install_wheel(
    uv: str,
    python: Path,
    wheel: Path,
    *,
    env: dict[str, str],
    scratch: Path,
    logs_dir: Path,
    label: str,
) -> None:
    run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        cwd=scratch,
        env=env,
        logs_dir=logs_dir,
        label=f"{label}-install",
    )



@contextmanager
def without_developer_target(target: Path) -> Iterator[None]:
    if target.is_symlink() or not target.is_dir():
        raise SmokeError(f"wheel smoke target is not a real directory: {target}")
    withheld = target.with_name(f".{target.name}-withheld")
    if withheld.exists() or withheld.is_symlink():
        raise SmokeError(f"cannot withhold target; destination already exists: {withheld}")
    target.rename(withheld)
    try:
        if target.exists():
            raise SmokeError(f"developer target remained accessible: {target}")
        yield
    finally:
        if target.exists() or target.is_symlink():
            raise SmokeError(f"developer target was recreated; cache retained at {withheld}")
        withheld.rename(target)

def run_consumer(
    *,
    python: Path,
    driver: Path,
    mode: str,
    work_root: Path,
    scratch: Path,
    compiler: Path,
    result_path: Path,
    logs_dir: Path,
    label: str,
    source_root: Path | None = None,
    source_copy: Path | None = None,
    compile_result: Path | None = None,
    library: Path | None = None,
    artifact_dir: Path | None = None,
    expected_runtime_sha256: str | None = None,
    developer_target: Path | None = None,
) -> dict[str, object]:
    work_root.mkdir(parents=True, exist_ok=True)
    env = consumer_environment(
        checkout_root=REPO_ROOT, compiler=compiler, cwd=work_root, scratch=scratch
    )
    command = consumer_command(
        python=python,
        driver=driver,
        mode=mode,
        work_root=work_root,
        checkout_root=REPO_ROOT,
        result_path=result_path,
        source_root=source_root,
        source_copy=source_copy,
        compile_result=compile_result,
        library=library,
        artifact_dir=artifact_dir,
        expected_runtime_sha256=expected_runtime_sha256,
        developer_target=developer_target,
    )
    if mode != "editable" and sys.platform == "darwin":
        sandbox = Path("/usr/bin/sandbox-exec")
        if not sandbox.is_file() or not (REPO_ROOT / "Cargo.toml").is_file():
            raise SmokeError("cannot prove original checkout denial without sandbox and checkout")
        policy = (
            "(version 1) (allow default) "
            f"(deny file-read* (subpath {json.dumps(str(REPO_ROOT))}))"
        )
        command = [str(sandbox), "-p", policy, *command, "--require-checkout-denied"]
    guard = without_developer_target(developer_target) if developer_target else nullcontext()
    try:
        with guard:
            completed = run(
                command,
                cwd=work_root,
                env=env,
                logs_dir=logs_dir,
                label=label,
            )
    except SmokeError as error:
        candidate = artifact_dir
        if candidate is None and mode in {"editable", "compile"}:
            candidate = work_root / "artifacts"
        if candidate is None and compile_result is not None and compile_result.is_file():
            metadata = read_json(compile_result)
            candidate = Path(metadata["artifact_dir"])
        if candidate is not None and candidate.is_dir():
            snapshot = logs_dir / f"{label}-artifacts"
            try:
                shutil.copytree(candidate, snapshot, symlinks=True)
            except OSError as snapshot_error:
                raise SmokeError(
                    f"{error}; artifact snapshot failed: {snapshot_error}"
                ) from error
            raise SmokeError(f"{error}; artifact snapshot: {snapshot}") from error
        raise
    try:
        evidence = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SmokeError(
            f"{label} consumer did not emit JSON: {completed.stdout!r}"
        ) from error
    if evidence.get("schema") != "chelis-python-wheel-consumer/2":
        raise SmokeError(f"{label} consumer emitted unexpected evidence: {evidence!r}")
    return evidence


def build_wheel(
    *,
    uv: str,
    source_root: Path,
    wheelhouse: Path,
    target: Path,
    env: dict[str, str],
    logs_dir: Path,
    label: str,
) -> Path:
    if target.is_relative_to(source_root) or source_root.is_relative_to(target):
        raise SmokeError("CARGO_TARGET_DIR must be separate from the disposable source copy")
    wheelhouse.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(target).free
    print(f"{label}: free bytes before wheel build: {before}", file=sys.stderr)
    run(
        [uv, "build", "--wheel", "--out-dir", str(wheelhouse), "bindings/python"],
        cwd=source_root,
        env=env,
        logs_dir=logs_dir,
        label=label,
    )
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        raise SmokeError(f"expected one {label} wheel, found {[wheel.name for wheel in wheels]}")
    return wheels[0]


def record_build_space(state: dict[str, object], target: Path, label: str) -> None:
    builds = state.setdefault("builds", [])
    if not isinstance(builds, list):
        raise SmokeError("internal receipt build list is malformed")
    builds.append(
        {
            "operation": label,
            "cargo_target_dir": str(target),
            "free_bytes_before": shutil.disk_usage(target).free,
        }
    )


def install_editable(
    *,
    uv: str,
    python: Path,
    source_root: Path,
    env: dict[str, str],
    logs_dir: Path,
) -> None:
    editable_env = dict(env)
    editable_env["VIRTUAL_ENV"] = str(python.parent.parent)
    editable_env["PATH"] = os.pathsep.join(
        (str(python.parent), editable_env.get("PATH", ""))
    )
    run(
        [uv, "pip", "install", "-e", "bindings/python"],
        cwd=source_root,
        env=editable_env,
        logs_dir=logs_dir,
        label="editable-install",
    )


def patch_cross_runtime_source(source_root: Path) -> None:
    path = source_root / RUNTIME_SOURCE
    original = path.read_bytes()
    if original.count(CROSS_BUILD_MARKER) != 1:
        raise SmokeError(f"expected one wheel-B marker in {RUNTIME_SOURCE}")
    path.write_bytes(original.replace(CROSS_BUILD_MARKER, CROSS_BUILD_REPLACEMENT, 1))


def run_key_seed_witness(
    *,
    compiler: Path,
    artifact_dir: Path,
    runtime_sha256: str,
    work_root: Path,
    expected_bits: int,
    env: dict[str, str],
    logs_dir: Path,
    label: str,
) -> dict[str, object]:
    """Link the staged archive, then observe a real exported runtime operation."""
    archive = artifact_dir / "libchelis_runtime.a"
    if sha256(archive) != runtime_sha256:
        raise SmokeError(f"{label} native witness archive differs from the wheel receipt")
    work_root.mkdir(parents=True, exist_ok=True)
    source = work_root / "key_seed.c"
    source.write_text(
        r"""#include "chelis_runtime.h"
#include <inttypes.h>
#include <stdio.h>

int main(void) {
    chelis_key key = chelis_key_from_seed(INT64_C(7));
    printf("%" PRIu64 "\n", key.bits);
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = work_root / "key_seed"
    run(
        [
            str(compiler),
            "-std=c11",
            "-I",
            str(artifact_dir),
            str(source),
            str(archive),
            "-lm",
            "-o",
            str(executable),
        ],
        cwd=work_root,
        env=env,
        logs_dir=logs_dir,
        label=f"{label}-link",
    )
    observed = run(
        [str(executable)],
        cwd=work_root,
        env=env,
        logs_dir=logs_dir,
        label=f"{label}-execute",
    ).stdout.strip()
    if observed != str(expected_bits):
        raise SmokeError(
            f"{label} returned key bits {observed!r}, expected {expected_bits}"
        )
    return {
        "seed": 7,
        "expected_bits": expected_bits,
        "observed_bits": int(observed),
        "runtime_archive_sha256": runtime_sha256,
    }


def unique_diagnostics_path(target: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return target / "wheel-smoke-failures" / f"{timestamp}-{time.time_ns()}"


def retain_diagnostics(
    scratch: Path, target: Path, state: dict[str, object], receipt_path: Path | None
) -> Path:
    destination = unique_diagnostics_path(target)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "receipts", "wheelhouse", "cross-wheelhouse"):
        source = scratch / name
        if source.is_dir():
            shutil.move(str(source), destination / name)
    state["diagnostics_path"] = str(destination)
    if receipt_path is not None:
        write_json(receipt_path, state)
        shutil.copy2(receipt_path, destination / "wheel-smoke.json")
    else:
        write_json(destination / "wheel-smoke.json", state)
    return destination


def initial_receipt(target: Path, source_build_mode: str) -> dict[str, object]:
    return {
        "schema": "chelis-python-wheel-smoke/2",
        "status": "running",
        "source_build_mode": source_build_mode,
        "cargo_target_dir": str(target),
        "runtime_mode": None,
        "runtime_archive_sha256": None,
        "library_sha256": None,
        "observed_value": None,
        "persisted_reload_value": None,
        "negative_controls": {},
        "process_results": {},
        "failures": [],
    }


@contextmanager
def temporary_smoke_directory(target: Path, common_root: Path) -> Iterator[Path]:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if temporary_root.is_relative_to(common_root):
        raise SmokeError(
            "temporary source copies cannot be created under the shared checkout: "
            f"{temporary_root}"
        )
    identity = hashlib.sha256(f"{REPO_ROOT}\0{target}".encode()).hexdigest()[:16]
    scratch = temporary_root / f"chelis-python-wheel-smoke-{identity}"
    if scratch.is_relative_to(common_root):
        raise SmokeError(f"temporary source copy entered the shared checkout: {scratch}")
    try:
        scratch.mkdir()
    except FileExistsError as error:
        raise SmokeError(
            f"wheel smoke scratch path already exists; inspect before removing: {scratch}"
        ) from error
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch)


def build_and_exercise(
    *, receipt_path: Path | None, wheel_path: Path | None, crossed_bundle: bool
) -> dict[str, object]:
    uv = shutil.which("uv")
    if uv is None:
        raise SmokeError(
            "uv is required; install the project-managed Python toolchain first"
        )
    compiler = compiler_path()
    if compiler.is_relative_to(REPO_ROOT):
        raise SmokeError(f"native compiler is inside the checkout: {compiler}")
    target = target_directory()
    target.mkdir(parents=True, exist_ok=True)
    env = build_environment(target)
    state = initial_receipt(target, "prebuilt-wheel" if wheel_path else "disposable-source-copy")
    if receipt_path is not None:
        write_json(receipt_path, state)

    common_root = common_repository_root()
    with temporary_smoke_directory(target, common_root) as scratch:
        logs_dir = scratch / "logs"
        receipts_dir = scratch / "receipts"
        wheelhouse = scratch / "wheelhouse"
        cross_wheelhouse = scratch / "cross-wheelhouse"
        relocated = scratch / "relocated"
        driver = scratch / "python_wheel_consumer.py"
        shutil.copy2(CONSUMER_SCRIPT, driver)
        for path in (logs_dir, receipts_dir, scratch / "home", scratch / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        editable_source = editable_source_directory(target, scratch)
        editable_source_owned = False

        frozen_wheel: Path | None = None
        if wheel_path is not None:
            supplied_wheel = wheel_path.resolve()
            if not supplied_wheel.is_file() or supplied_wheel.suffix != ".whl":
                raise SmokeError(f"prebuilt wheel does not exist: {supplied_wheel}")
            wheelhouse.mkdir(parents=True, exist_ok=True)
            frozen_wheel = wheelhouse / supplied_wheel.name
            shutil.copy2(supplied_wheel, frozen_wheel)
        try:
            # The editable invocation exercises the normal development feature set.
            copy_source_tree(editable_source)
            editable_source_owned = True

            editable_venv_dir = scratch / "editable-venv"
            editable_python = make_venv(
                uv, editable_venv_dir, env=env, scratch=scratch, logs_dir=logs_dir, label="editable"
            )
            record_build_space(state, target, "uv pip install -e bindings/python")
            state["editable_probe"] = {
                "invocation": "uv pip install -e bindings/python",
                "status": "building",
                "source_copy": str(editable_source),
            }
            if receipt_path is not None:
                write_json(receipt_path, state)
            refresh_editable_runtime_bundle(
                target=target,
                source_root=editable_source,
                env=env,
                logs_dir=logs_dir,
                label="editable-runtime-bundle-refresh",
            )

            install_editable(
                uv=uv,
                python=editable_python,
                source_root=editable_source,
                env=env,
                logs_dir=logs_dir,
            )
            editable_work = relocated / "editable-consumer"
            editable_result_path = receipts_dir / "editable-consumer.json"
            editable_result = run_consumer(
                python=editable_python,
                driver=driver,
                mode="editable",
                work_root=editable_work,
                scratch=scratch,
                compiler=compiler,
                result_path=editable_result_path,
                logs_dir=logs_dir,
                label="editable-consumer",
                source_root=editable_source,
            )
            if editable_result.get("runtime_mode") != "development":
                raise SmokeError("editable installation did not stage development runtime")
            if not editable_result.get("freshness_mutation", {}).get("rejected"):
                raise SmokeError("editable runtime-source mutation was not rejected")
            state["editable_probe"] = {
                "invocation": "uv pip install -e bindings/python",
                "runtime_mode": editable_result["runtime_mode"],
                "runtime_archive_sha256": editable_result["runtime_archive_sha256"],
                "observed_value": editable_result["observed_value"],
                "freshness_mutation": editable_result["freshness_mutation"],
                "source_copy_retained_for_wheel_build": wheel_path is None,
            }
            write_json(receipts_dir / "editable-probe.json", state["editable_probe"])
            if receipt_path is not None:
                write_json(receipt_path, state)

            if wheel_path is None:
                wheel_source = editable_source
                if not wheel_source.is_dir():
                    raise SmokeError("editable source copy disappeared before wheel build")
                if target.is_relative_to(wheel_source):
                    raise SmokeError("CARGO_TARGET_DIR is inside the disposable wheel source")
                wheel_build_receipt = {
                    "mode": "standard-uv-build-from-disposable-source-copy",
                    "invocation": "uv build --wheel --out-dir <scratch-wheelhouse> bindings/python",
                    "source_copy": str(wheel_source),
                    "source_copy_absent_before_consumer": False,
                    "status": "building",
                }
                state["wheel_build"] = wheel_build_receipt
                record_build_space(state, target, "uv build --wheel bindings/python")
                if receipt_path is not None:
                    write_json(receipt_path, state)
                built_wheel = build_wheel(
                    uv=uv,
                    source_root=wheel_source,
                    wheelhouse=wheelhouse,
                    target=target,
                    env=env,
                    logs_dir=logs_dir,
                    label="sealed-wheel-build",
                )
                shutil.rmtree(wheel_source)
                if wheel_source.exists():
                    raise SmokeError("disposable wheel source copy was not removed")
                source_copy_for_wheel: Path | None = wheel_source
                wheel_build_receipt.update(
                    {
                        "source_copy_absent_before_consumer": True,
                        "status": "built",
                        "wheel_filename": built_wheel.name,
                    }
                )
            else:
                shutil.rmtree(editable_source)
                if editable_source.exists():
                    raise SmokeError("editable source copy was not removed before wheel install")
                if frozen_wheel is None:
                    raise SmokeError("prebuilt wheel was not preserved before editable install")
                built_wheel = frozen_wheel
                source_copy_for_wheel = None
                wheel_build_receipt = {
                    "mode": "prebuilt-wheel",
                    "source_free_build_proven": False,
                    "wheel_input": str(supplied_wheel),
                    "status": "supplied",
                }
                state["wheel_build"] = wheel_build_receipt
            if receipt_path is not None:
                write_json(receipt_path, state)

            installed_wheel = relocated / "wheelhouse" / built_wheel.name
            installed_wheel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_wheel, installed_wheel)
            wheel_sha256 = sha256(installed_wheel)
            wheel_build_receipt["wheel_sha256"] = wheel_sha256
            state.update(
                {
                    "source_free_build_proven": False,
                    "wheel_filename": installed_wheel.name,
                    "wheel_sha256": wheel_sha256,
                }
            )
            if receipt_path is not None:
                write_json(receipt_path, state)
            wheel_venv_dir = relocated / "wheel-venv"
            wheel_python = make_venv(
                uv, wheel_venv_dir, env=env, scratch=scratch, logs_dir=logs_dir, label="wheel"
            )
            install_wheel(
                uv,
                wheel_python,
                installed_wheel,
                env=env,
                scratch=scratch,
                logs_dir=logs_dir,
                label="sealed-wheel",
            )

            wheel_work = relocated / "wheel-consumer"
            compile_result_path = receipts_dir / "wheel-compile.json"
            compile_result = run_consumer(
                python=wheel_python,
                driver=driver,
                mode="compile",
                work_root=wheel_work,
                scratch=scratch,
                compiler=compiler,
                result_path=compile_result_path,
                logs_dir=logs_dir,
                label="sealed-wheel-compile-and-call",
                developer_target=target,
                source_copy=source_copy_for_wheel,
            )
            if compile_result.get("runtime_mode") != "sealed":
                raise SmokeError("standard wheel did not stage a sealed runtime")
            if source_copy_for_wheel is not None and source_copy_for_wheel.exists():
                raise SmokeError("disposable source copy exists during wheel execution")
            native_witness = run_key_seed_witness(
                compiler=compiler,
                artifact_dir=Path(compile_result["artifact_dir"]),
                runtime_sha256=str(compile_result["runtime_archive_sha256"]),
                work_root=relocated / "wheel-key-witness",
                expected_bits=7,
                env=env,
                logs_dir=logs_dir,
                label="sealed-wheel-key-witness",
            )
            state.update(
                {
                    "runtime_mode": compile_result["runtime_mode"],
                    "runtime_archive_sha256": compile_result["runtime_archive_sha256"],
                    "library_sha256": compile_result["library_sha256"],
                    "observed_value": compile_result["observed_value"],
                    "native_runtime_witness": native_witness,
                    "negative_controls": compile_result["negative_controls"],
                    "process_results": {"compile_and_call": compile_result},
                }
            )
            if receipt_path is not None:
                write_json(receipt_path, state)

            source_file = Path(compile_result["source_file"])
            source_file.unlink()
            source_file.parent.rmdir()
            if source_file.exists():
                raise SmokeError("original source file still exists after compile process")
            reload_result_path = receipts_dir / "wheel-reload.json"
            reload_result = run_consumer(
                python=wheel_python,
                driver=driver,
                mode="reload",
                work_root=wheel_work,
                scratch=scratch,
                compiler=compiler,
                result_path=reload_result_path,
                logs_dir=logs_dir,
                label="sealed-wheel-persisted-reload",
                developer_target=target,
                source_copy=source_copy_for_wheel,
                compile_result=compile_result_path,
            )
            if reload_result.get("runtime_mode") != "sealed":
                raise SmokeError("persisted reload did not admit a sealed artifact")
            if reload_result.get("original_source_absent") is not True:
                raise SmokeError("persisted reload ran while original source remained")
            state["persisted_reload_value"] = reload_result["observed_value"]
            process_results = state.get("process_results")
            if not isinstance(process_results, dict):
                raise SmokeError("internal receipt process result is malformed")
            process_results["persisted_reload"] = reload_result
            consumer_results = [compile_result, reload_result]
            write_json(receipts_dir / "wheel-success.json", state)
            if receipt_path is not None:
                write_json(receipt_path, state)

            if crossed_bundle:
                crossed_source = editable_source
                copy_source_tree(crossed_source)
                patch_cross_runtime_source(crossed_source)
                refresh_editable_runtime_bundle(
                    target=target,
                    source_root=crossed_source,
                    env=env,
                    logs_dir=logs_dir,
                    label="crossed-runtime-bundle-refresh",
                )
                crossed_receipt: dict[str, object] = {
                    "status": "building",
                    "source_copy": str(crossed_source),
                    "source_copy_absent_before_consumer": False,
                    "wheel_a_sha256": wheel_sha256,
                    "runtime_a_sha256": compile_result["runtime_archive_sha256"],
                }
                state["crossed_bundle"] = crossed_receipt
                record_build_space(state, target, "uv build --wheel crossed runtime")
                if receipt_path is not None:
                    write_json(receipt_path, state)
                crossed_wheel = build_wheel(
                    uv=uv,
                    source_root=crossed_source,
                    wheelhouse=cross_wheelhouse,
                    target=target,
                    env=env,
                    logs_dir=logs_dir,
                    label="crossed-sealed-wheel-build",
                )
                shutil.rmtree(crossed_source)
                if crossed_source.exists():
                    raise SmokeError("crossed-wheel source copy was not removed")
                crossed_receipt.update(
                    {
                        "status": "built",
                        "source_copy_absent_before_consumer": True,
                        "wheel_b_sha256": sha256(crossed_wheel),
                    }
                )
                if receipt_path is not None:
                    write_json(receipt_path, state)

                crossed_venv_dir = relocated / "crossed-wheel-venv"
                crossed_python = make_venv(
                    uv,
                    crossed_venv_dir,
                    env=env,
                    scratch=scratch,
                    logs_dir=logs_dir,
                    label="crossed-wheel",
                )
                install_wheel(
                    uv,
                    crossed_python,
                    crossed_wheel,
                    env=env,
                    scratch=scratch,
                    logs_dir=logs_dir,
                    label="crossed-wheel",
                )
                crossed_work = relocated / "crossed-consumer"
                crossed_compile_path = receipts_dir / "crossed-wheel-compile.json"
                crossed_compile = run_consumer(
                    python=crossed_python,
                    driver=driver,
                    mode="compile",
                    work_root=crossed_work,
                    scratch=scratch,
                    compiler=compiler,
                    result_path=crossed_compile_path,
                    logs_dir=logs_dir,
                    label="crossed-wheel-compile-and-call",
                    developer_target=target,
                    source_copy=crossed_source,
                )
                crossed_witness = run_key_seed_witness(
                    compiler=compiler,
                    artifact_dir=Path(crossed_compile["artifact_dir"]),
                    runtime_sha256=str(crossed_compile["runtime_archive_sha256"]),
                    work_root=relocated / "crossed-key-witness",
                    expected_bits=8,
                    env=env,
                    logs_dir=logs_dir,
                    label="crossed-wheel-key-witness",
                )
                crossed_receipt.update(
                    {
                        "runtime_b_sha256": crossed_compile["runtime_archive_sha256"],
                        "library_b_sha256": crossed_compile["library_sha256"],
                        "observed_value_b": crossed_compile["observed_value"],
                        "native_runtime_witness": crossed_witness,
                        "compile_process": crossed_compile,
                    }
                )
                if receipt_path is not None:
                    write_json(receipt_path, state)
                if crossed_compile["runtime_archive_sha256"] == compile_result["runtime_archive_sha256"]:
                    raise SmokeError("the second sealed wheel did not carry distinct runtime bytes")
                crossed_source_file = Path(crossed_compile["source_file"])
                crossed_source_file.unlink()
                crossed_source_file.parent.rmdir()
                cross_result_path = receipts_dir / "crossed-wheel-rejection.json"
                cross_result = run_consumer(
                    python=wheel_python,
                    driver=driver,
                    mode="crossed-load",
                    work_root=crossed_work,
                    scratch=scratch,
                    compiler=compiler,
                    result_path=cross_result_path,
                    logs_dir=logs_dir,
                    label="crossed-sealed-wheel-rejection",
                    developer_target=target,
                    source_copy=crossed_source,
                    library=Path(crossed_compile["library_path"]),
                    artifact_dir=Path(crossed_compile["artifact_dir"]),
                    expected_runtime_sha256=str(compile_result["runtime_archive_sha256"]),
                )
                crossed_receipt.update({"status": "rejected", "rejection": cross_result})
                consumer_results.extend((crossed_compile, cross_result))
                if receipt_path is not None:
                    write_json(receipt_path, state)

            # Copy removal alone is not proof: on hosts without checkout
            # denial the installed consumer might still read the checkout.
            state["source_free_build_proven"] = wheel_path is None and all(
                result["environment"]["checkout_read_denied"] is True
                for result in consumer_results
            )
            state["status"] = "passed"
            state["failures"] = []
            if receipt_path is not None:
                write_json(receipt_path, state)
            return state
        except Exception as error:
            state["status"] = "failed"
            failures = state.setdefault("failures", [])
            if isinstance(failures, list):
                failures.append({"type": type(error).__name__, "message": str(error)})
            diagnostics = retain_diagnostics(scratch, target, state, receipt_path)
            raise SmokeError(f"{error}\nDiagnostics retained at {diagnostics}") from error
        finally:
            if editable_source_owned and editable_source.exists():
                shutil.rmtree(editable_source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="write structured evidence (also retained with diagnostics on failure)",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="consume an existing wheel instead of building; does not prove its build was source-free",
    )
    parser.add_argument(
        "--crossed-bundle",
        action="store_true",
        help="build a second runtime-distinct sealed wheel and test crossed-wheel rejection",
    )
    args = parser.parse_args()
    receipt_path = args.receipt.resolve() if args.receipt else None
    wheel_path = args.wheel.resolve() if args.wheel else None
    print("Python wheel smoke starting", file=sys.stderr, flush=True)
    try:
        evidence = build_and_exercise(
            receipt_path=receipt_path,
            wheel_path=wheel_path,
            crossed_bundle=args.crossed_bundle,
        )
    except Exception as error:
        print(f"Python wheel smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
