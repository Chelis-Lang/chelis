#!/usr/bin/env python3
"""Exercise the repository-owned Kache wrapper and its fallback control."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


class ContractFailure(RuntimeError):
    """The active shell does not satisfy the managed Kache contract."""


@dataclass(frozen=True)
class ManagedKache:
    wrapper: Path
    config: Path


def validate_environment(
    repo_root: Path,
    environ: Mapping[str, str],
    discovered_wrapper: Path | None,
) -> ManagedKache:
    raw_wrapper = environ.get("RUSTC_WRAPPER")
    if not raw_wrapper:
        raise ContractFailure("RUSTC_WRAPPER is not set")
    wrapper = Path(raw_wrapper)
    if not wrapper.is_file():
        raise ContractFailure(f"RUSTC_WRAPPER is not a file: {wrapper}")
    if discovered_wrapper is None:
        raise ContractFailure("PATH does not expose the managed kache binary")
    if not discovered_wrapper.is_file() or not wrapper.samefile(discovered_wrapper):
        raise ContractFailure(
            f"PATH kache differs from RUSTC_WRAPPER: {discovered_wrapper} != {wrapper}"
        )

    raw_config = environ.get("KACHE_CONFIG")
    if not raw_config:
        raise ContractFailure("KACHE_CONFIG is not set")
    config = Path(raw_config)
    expected_config = repo_root / ".kache.toml"
    if not config.is_file() or not expected_config.is_file() or not config.samefile(
        expected_config
    ):
        raise ContractFailure(
            f"KACHE_CONFIG must select the repository policy: {raw_config}"
        )
    if environ.get("KACHE_DISABLED", "0").lower() not in {"0", "false"}:
        raise ContractFailure("KACHE_DISABLED must be forced off in Devenv")

    with config.open("rb") as stream:
        cache = tomllib.load(stream).get("cache", {})
    required = {
        "ignore_env": True,
        "cache_executables": True,
        "heartbeat_secs": 30,
    }
    if any(cache.get(key) != value for key, value in required.items()):
        raise ContractFailure(f"repository Kache policy differs from {required!r}")
    return ManagedKache(wrapper=wrapper, config=config)


def run_checked(
    command: list[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractFailure(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def write_probe_project(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "Cargo.toml").write_text(
        '[package]\nname = "kache-smoke"\nversion = "0.0.0"\n'
        'edition = "2024"\n',
        encoding="utf-8",
    )
    (root / "src/main.rs").write_text(
        'fn main() { println!("managed kache smoke"); }\n',
        encoding="utf-8",
    )


def main() -> int:
    discovered = shutil.which("kache")
    managed = validate_environment(
        REPO_ROOT,
        os.environ,
        Path(discovered) if discovered else None,
    )
    version = run_checked(
        [str(managed.wrapper), "--version"],
        environment=os.environ,
        cwd=REPO_ROOT,
    )
    if "kache 0.16.0" not in version.lower():
        raise ContractFailure(f"unexpected managed Kache version: {version.strip()}")

    cargo = shutil.which("cargo")
    if cargo is None:
        raise ContractFailure("managed Cargo is absent from PATH")

    with tempfile.TemporaryDirectory(prefix="chelis-kache-smoke-") as raw_directory:
        temporary = Path(raw_directory)
        project = temporary / "probe"
        write_probe_project(project)
        home = temporary / "home"
        cargo_home = temporary / "cargo-home"
        home.mkdir()
        cargo_home.mkdir()
        (cargo_home / "config.toml").write_text(
            '[build]\nrustc-wrapper = "/definitely/missing/host-kache"\n',
            encoding="utf-8",
        )
        common = dict(os.environ)
        common.update(
            {
                "HOME": str(home),
                "XDG_CACHE_HOME": str(temporary / "xdg-cache"),
                "CARGO_HOME": str(cargo_home),
                "CARGO_INCREMENTAL": "0",
            }
        )

        hostile = dict(common)
        hostile["CARGO_TARGET_DIR"] = str(temporary / "managed-target")
        run_checked([cargo, "check"], environment=hostile, cwd=project)

        no_cache_home = temporary / "no-cache-cargo-home"
        no_cache_home.mkdir()
        control = dict(common)
        control.pop("RUSTC_WRAPPER", None)
        control.pop("CARGO_BUILD_RUSTC_WRAPPER", None)
        control.pop("KACHE_DISABLED", None)
        control["CARGO_HOME"] = str(no_cache_home)
        control["CARGO_TARGET_DIR"] = str(temporary / "no-cache-target")
        run_checked([cargo, "check"], environment=control, cwd=project)

        (cargo_home / "config.toml").write_text(
            f'[build]\nrustc-wrapper = "{managed.wrapper}"\n',
            encoding="utf-8",
        )
        doctor = run_checked(
            [str(managed.wrapper), "doctor"],
            environment=common,
            cwd=project,
        )
        if "All checks passed." not in doctor:
            raise ContractFailure(f"healthy isolated doctor result was unclear:\n{doctor}")

    print("managed Kache smoke: PASS (pinned wrapper, hostile config, no-cache control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
