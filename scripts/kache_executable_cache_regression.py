#!/usr/bin/env python3
"""Benchmark cross-worktree test-executable caching under managed Kache."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from kache_toolchain_smoke import (
    REPO_ROOT,
    validate_environment,
    write_isolated_kache_config,
)


class ReportFailure(RuntimeError):
    """The warm Kache report does not prove the executable-cache contract."""


CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_KACHE_KEY_SCHEMA = 28
MACH_O_UUID_PATTERN = re.compile(
    r"^UUID:\s*([0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})(?:\s|$)",
    re.MULTILINE,
)


def isolated_environment(
    base: Mapping[str, str],
    *,
    target: Path,
    config: Path,
) -> dict[str, str]:
    """Give one build its own target and both builds one fresh Kache store."""

    environment = dict(base)
    environment["CARGO_TARGET_DIR"] = str(target)
    environment["KACHE_CONFIG"] = str(config)
    environment["KACHE_LOG_FILE"] = "kache=debug"
    environment["KACHE_LOG_FILE_PATH"] = str(config.parent / "kache-wrapper.log")
    return environment


def checkout_target(checkout: Path) -> Path:
    """Keep Cargo's target beneath the clone so Kache can verify its workspace."""

    return checkout / "target"


def kache_store(cache_dir: Path) -> Path:
    """Resolve Kache's store beneath an explicit probe-local cache directory."""

    return cache_dir / "store"


def validate_warm_report(report: dict[str, Any]) -> None:
    local_hits = int(report.get("summary", {}).get("local_hits", 0))
    if local_hits == 0:
        raise ReportFailure("warm report contains no local hits")
    restored = int(report.get("storage", {}).get("restored_bytes", 0))
    if restored == 0:
        raise ReportFailure("warm report contains no restored bytes")
    reasons = report.get("bypass", {}).get("reasons", [])
    uncached = [
        reason
        for reason in reasons
        if "cache_executables=false" in str(reason.get("reason", ""))
    ]
    if uncached:
        raise ReportFailure(
            f"warm report contains executable bypass rows: {uncached!r}"
        )


def validate_legacy_namespace_report(report: dict[str, Any]) -> None:
    """Require a schema-28 consumer to compile past a schema-27-only store."""

    local_hits = int(report.get("summary", {}).get("local_hits", 0))
    events = report.get("all_events", [])
    reused = [
        event
        for event in events
        if isinstance(event, dict) and event.get("result") == "local_hit"
    ]
    if local_hits != 0 or reused:
        raise ReportFailure(
            "current Kache reused a schema-27 entry: "
            f"local_hits={local_hits}, events={reused!r}"
        )
    misses = int(report.get("summary", {}).get("misses", 0))
    miss_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("result") == "miss"
    ]
    if misses == 0 or not miss_events:
        raise ReportFailure(
            "schema-transition report contains no current-consumer misses"
        )


def run_timed(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> tuple[float, str]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise ReportFailure(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    summary = "\n".join(completed.stdout.rstrip().splitlines()[-3:])
    print(f"PASS ({elapsed:.3f}s): {' '.join(command)}")
    if summary:
        print(summary)
    return elapsed, completed.stdout


def cache_entry_schemas(store: Path) -> frozenset[int]:
    schemas: set[int] = set()
    for metadata_path in sorted(store.glob("*/meta.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReportFailure(
                f"cache metadata is unreadable: {metadata_path}: {error}"
            ) from error
        schema = metadata.get("key_schema") if isinstance(metadata, dict) else None
        if not isinstance(schema, int):
            raise ReportFailure(
                f"cache metadata has no integer key schema: {metadata_path}"
            )
        schemas.add(schema)
    return frozenset(schemas)


def run_schema_transition_probe(
    *,
    cargo: str,
    current_wrapper: Path,
    legacy_wrapper: Path,
    policy: str,
    temporary: Path,
) -> dict[str, Any]:
    """Prove an old producer misses under the new cache representation."""

    root = temporary / "schema-transition"
    seed = root / "seed"
    seed.mkdir(parents=True)
    (seed / "src").mkdir()
    (seed / "Cargo.toml").write_text(
        '[package]\nname = "kache_schema_probe"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (seed / "src/lib.rs").write_text(
        "pub fn answer() -> u8 { 42 }\n\n"
        "#[cfg(test)]\nmod tests {\n"
        "    #[test]\n    fn answer_is_stable() {\n"
        "        assert_eq!(super::answer(), 42);\n    }\n}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=seed, check=True)
    subprocess.run(["git", "add", "--all"], cwd=seed, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Kache schema probe",
            "-c",
            "user.email=probe@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "schema probe",
        ],
        cwd=seed,
        check=True,
    )
    checkouts = [root / "legacy", root / "current"]
    for checkout in checkouts:
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--local",
                "--no-hardlinks",
                str(seed),
                str(checkout),
            ],
            check=True,
        )

    cache_dir = root / "kache-cache"
    config = root / "kache.toml"
    write_isolated_kache_config(policy, config, cache_dir)
    base = dict(os.environ)
    base["HOME"] = str(root / "home")
    base["CARGO_HOME"] = str(root / "cargo-home")
    base["TMPDIR"] = str(root / "tmp")
    base["CARGO_INCREMENTAL"] = "0"
    base["KACHE_DISABLED"] = "0"
    base["RUSTC_WORKSPACE_WRAPPER"] = ""
    base["CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"] = ""
    isolated_directories = (
        Path(base["HOME"]),
        Path(base["CARGO_HOME"]),
        Path(base["TMPDIR"]),
    )
    for directory in isolated_directories:
        directory.mkdir(parents=True)

    environments: list[dict[str, str]] = []
    wrappers = (legacy_wrapper, current_wrapper)
    for checkout, wrapper in zip(checkouts, wrappers, strict=True):
        environment = isolated_environment(
            base,
            target=checkout_target(checkout),
            config=config,
        )
        environment["RUSTC_WRAPPER"] = str(wrapper)
        environment["CARGO_BUILD_RUSTC_WRAPPER"] = str(wrapper)
        environments.append(environment)

    command = [cargo, "test", "--no-run", "--offline"]
    legacy_s, _ = run_timed(command, cwd=checkouts[0], environment=environments[0])
    current_s, _ = run_timed(command, cwd=checkouts[1], environment=environments[1])
    report_output = subprocess.run(
        [
            str(current_wrapper),
            "report",
            "--format",
            "json",
            "--since",
            "1h",
            "--root",
            str(checkouts[1]),
        ],
        cwd=checkouts[1],
        env=environments[1],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    report = json.loads(report_output)
    validate_legacy_namespace_report(report)
    schemas = cache_entry_schemas(kache_store(cache_dir))
    if not {27, EXPECTED_KACHE_KEY_SCHEMA}.issubset(schemas):
        raise ReportFailure(
            "schema-transition store does not contain both producer formats: "
            f"schemas={sorted(schemas)!r}"
        )
    return {
        "legacy_producer_wall_s": legacy_s,
        "current_consumer_wall_s": current_s,
        "current_local_hits": 0,
        "current_misses": report["summary"]["misses"],
        "store_key_schemas": sorted(schemas),
    }


def executable_candidates(target: Path) -> list[Path]:
    dependencies = target / "debug" / "deps"
    if not dependencies.is_dir():
        return []
    return sorted(
        path
        for path in dependencies.iterdir()
        if path.is_file()
        and not path.suffix
        and stat.S_IMODE(path.stat().st_mode) & 0o111
    )


def restored_executable_candidates(
    candidates: list[Path], report: dict[str, Any], store: Path
) -> list[Path]:
    restored_names: set[str] = set()
    for event in report.get("all_events", []):
        if not isinstance(event, dict) or event.get("result") != "local_hit":
            continue
        cache_key = event.get("cache_key")
        crate_name = event.get("crate_name")
        if (
            not isinstance(cache_key, str)
            or CACHE_KEY_PATTERN.fullmatch(cache_key) is None
        ):
            raise ReportFailure(f"local-hit cache key is invalid: {cache_key!r}")
        if not isinstance(crate_name, str) or not crate_name:
            raise ReportFailure(f"local-hit crate name is invalid: {crate_name!r}")
        meta_path = store / cache_key / "meta.json"
        if not meta_path.is_file():
            raise ReportFailure(f"local-hit cache metadata is absent: {meta_path}")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReportFailure(
                f"local-hit cache metadata is unreadable: {meta_path}: {error}"
            ) from error
        if not isinstance(metadata, dict):
            raise ReportFailure(
                f"local-hit cache metadata is not an object: {meta_path}"
            )
        if metadata.get("cache_key") != cache_key:
            raise ReportFailure(f"local-hit cache metadata key mismatch: {meta_path}")
        if metadata.get("key_schema") != EXPECTED_KACHE_KEY_SCHEMA:
            raise ReportFailure(
                "local-hit cache metadata schema mismatch: "
                f"expected {EXPECTED_KACHE_KEY_SCHEMA}, "
                f"got {metadata.get('key_schema')!r}: {meta_path}"
            )
        if metadata.get("crate_name") != crate_name:
            raise ReportFailure(f"local-hit cache metadata crate mismatch: {meta_path}")
        files = metadata.get("files")
        if not isinstance(files, list):
            raise ReportFailure(
                f"local-hit cache metadata files are invalid: {meta_path}"
            )
        for artifact in files:
            if not isinstance(artifact, dict):
                raise ReportFailure(
                    f"local-hit cache metadata contains an invalid file row: {meta_path}"
                )
            name = artifact.get("name")
            executable = artifact.get("executable")
            if not isinstance(name, str) or not name:
                raise ReportFailure(
                    f"local-hit cache metadata contains an invalid file name: {meta_path}"
                )
            if not isinstance(executable, bool):
                raise ReportFailure(
                    "local-hit cache metadata contains an invalid executable flag: "
                    f"{meta_path}"
                )
            if executable:
                restored_names.add(name)
    restored = [path for path in candidates if path.name in restored_names]
    if candidates and not restored:
        candidate_crates = {
            path.name.rsplit("-", 1)[0].replace("-", "_") for path in candidates
        }
        candidate_events = [
            {
                "crate_name": event.get("crate_name"),
                "result": event.get("result"),
                "cache_key": event.get("cache_key"),
                "store_error": event.get("store_error"),
            }
            for event in report.get("all_events", [])
            if isinstance(event, dict) and event.get("crate_name") in candidate_crates
        ]
        raise ReportFailure(
            "warm local hits contain no exact executable metadata match: "
            f"candidates={sorted(path.name for path in candidates)!r}, "
            f"cached_executables={sorted(restored_names)!r}, "
            f"candidate_events={candidate_events!r}"
        )
    return restored


def validate_debug_uuid_outputs(
    binary_output: str, bundle_output: str
) -> frozenset[str]:
    """Require exact, nonempty Mach-O UUID equality for a binary and its dSYM."""

    binary_uuids = frozenset(
        match.group(1).upper() for match in MACH_O_UUID_PATTERN.finditer(binary_output)
    )
    bundle_uuids = frozenset(
        match.group(1).upper() for match in MACH_O_UUID_PATTERN.finditer(bundle_output)
    )
    if not binary_uuids:
        raise ReportFailure("restored executable has no UUID")
    if not bundle_uuids:
        raise ReportFailure("restored dSYM bundle has no UUID")
    if binary_uuids != bundle_uuids:
        raise ReportFailure(
            "restored executable and dSYM UUID mismatch: "
            f"binary={sorted(binary_uuids)!r}, bundle={sorted(bundle_uuids)!r}"
        )
    return binary_uuids


def read_debug_uuids(command_prefix: list[str], binary: Path, bundle: Path) -> None:
    """Run dwarfdump on both restored artifacts and validate their identity."""

    outputs: list[str] = []
    for label, path in (("executable", binary), ("dSYM bundle", bundle)):
        completed = subprocess.run(
            [*command_prefix, "--uuid", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise ReportFailure(
                f"dwarfdump cannot read restored {label} {path}:\n{completed.stdout}"
            )
        outputs.append(completed.stdout)
    validate_debug_uuid_outputs(*outputs)


def contains_bytes(path: Path, needle: bytes, chunk_size: int = 1024 * 1024) -> bool:
    if not needle:
        raise ValueError("binary scan needle must not be empty")
    carry = b""
    overlap = len(needle) - 1
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            window = carry + chunk
            if needle in window:
                return True
            carry = window[-overlap:] if overlap else b""
    return False


def debug_bundle_files(binary: Path) -> list[Path]:
    bundle = Path(f"{binary}.dSYM")
    if not bundle.is_dir():
        return []
    return sorted(path for path in bundle.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["chelis-compiler-api", "chelis-cli"],
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "target/oracles/kache-executable-cache.json",
    )
    arguments = parser.parse_args()
    discovered = shutil.which("kache")
    managed = validate_environment(
        REPO_ROOT,
        os.environ,
        Path(discovered) if discovered else None,
    )
    cargo = shutil.which("cargo")
    if cargo is None:
        raise ReportFailure("Cargo is absent from the managed shell")
    legacy_wrapper_value = os.environ.get("KACHE_SCHEMA_27_WRAPPER")
    if not legacy_wrapper_value:
        raise ReportFailure("managed shell does not expose KACHE_SCHEMA_27_WRAPPER")
    legacy_wrapper = Path(legacy_wrapper_value)
    if not legacy_wrapper.is_file() or not os.access(legacy_wrapper, os.X_OK):
        raise ReportFailure(
            f"schema-27 Kache fixture is not executable: {legacy_wrapper}"
        )
    if legacy_wrapper.samefile(managed.wrapper):
        raise ReportFailure("schema-27 fixture resolves to the current Kache wrapper")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    package_flags = [flag for package in arguments.packages for flag in ("-p", package)]
    suite = [cargo, "nextest", "run", *package_flags, "--no-fail-fast"]

    with tempfile.TemporaryDirectory(prefix="chelis-kache-executables-") as raw:
        temporary = Path(raw)
        policy = managed.config.read_text(encoding="utf-8")
        schema_transition = run_schema_transition_probe(
            cargo=cargo,
            current_wrapper=managed.wrapper,
            legacy_wrapper=legacy_wrapper,
            policy=policy,
            temporary=temporary,
        )
        checkouts = [temporary / "cold", temporary / "warm"]
        for checkout in checkouts:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--local",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(REPO_ROOT),
                    str(checkout),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", "--detach", head],
                cwd=checkout,
                check=True,
            )

        targets = [checkout_target(checkout) for checkout in checkouts]
        cache_dir = temporary / "kache-cache"
        probe_config = temporary / "kache.toml"
        write_isolated_kache_config(
            policy,
            probe_config,
            cache_dir,
        )
        environments = [
            isolated_environment(
                os.environ,
                target=target,
                config=probe_config,
            )
            for target in targets
        ]

        cold_s, _ = run_timed(suite, cwd=checkouts[0], environment=environments[0])
        warm_s, _ = run_timed(suite, cwd=checkouts[1], environment=environments[1])
        report_output = subprocess.run(
            [
                str(managed.wrapper),
                "report",
                "--format",
                "json",
                "--since",
                "1h",
                "--root",
                str(checkouts[1]),
            ],
            cwd=checkouts[1],
            env=environments[1],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        report = json.loads(report_output)
        validate_warm_report(report)

        candidates = executable_candidates(targets[1])
        if not candidates:
            raise ReportFailure("warm suite produced no executable test binaries")
        try:
            restored_candidates = restored_executable_candidates(
                candidates, report, kache_store(cache_dir)
            )
        except ReportFailure as error:
            wrapper_log = temporary / "kache-wrapper.log"
            diagnostics = (
                wrapper_log.read_text(encoding="utf-8", errors="replace").splitlines()
                if wrapper_log.is_file()
                else []
            )
            relevant = [
                line
                for line in diagnostics
                if "not caching" in line
                or "failed to package" in line
                or "strip" in line
            ]
            suffix = "\n".join(relevant[-100:])
            if suffix:
                raise ReportFailure(
                    f"{error}\nKache wrapper diagnostics:\n{suffix}"
                ) from error
            raise
        if not restored_candidates:
            raise ReportFailure(
                "warm report contains no local hit for an executable test binary"
            )
        cold_path = str(checkouts[0]).encode()
        artifact_files = [
            artifact
            for binary in restored_candidates
            for artifact in (binary, *debug_bundle_files(binary))
        ]
        leaked = [path for path in artifact_files if contains_bytes(path, cold_path)]
        if leaked:
            raise ReportFailure(
                f"restored executables retain the cold checkout path: {leaked}"
            )

        debugger = "not-applicable"
        if sys_platform_is_darwin():
            dwarfdump = shutil.which("dwarfdump")
            xcrun = shutil.which("xcrun")
            if dwarfdump is None and xcrun is None:
                raise ReportFailure("Darwin debugger metadata checker is absent")
            command_prefix = (
                [dwarfdump]
                if dwarfdump is not None
                else [xcrun or "xcrun", "dwarfdump"]
            )
            for binary in restored_candidates:
                bundle = Path(f"{binary}.dSYM")
                if not bundle.is_dir():
                    raise ReportFailure(
                        f"restored Darwin test executable has no dSYM bundle: {binary}"
                    )
                read_debug_uuids(command_prefix, binary, bundle)
            debugger = (
                "restored and UUID-matched "
                f"{len(restored_candidates)} executable/dSYM pair(s)"
            )

        result = {
            "commit": head,
            "packages": arguments.packages,
            "cold_wall_s": cold_s,
            "warm_wall_s": warm_s,
            "warm_over_cold": warm_s / cold_s if cold_s else None,
            "local_hits": report["summary"]["local_hits"],
            "restored_bytes": report["storage"]["restored_bytes"],
            "warm_test_executables": len(candidates),
            "restored_test_executables": len(restored_candidates),
            "debugger": debugger,
            "cold_checkout_path_leaks": 0,
            "schema_transition": schema_transition,
        }
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    return 0


def sys_platform_is_darwin() -> bool:
    return os.uname().sysname == "Darwin"


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        raise SystemExit(main())
