#!/usr/bin/env python3
"""Execute old/current cache compatibility for checked runtime extents.

The previous producer must be a real pre-change chelis executable. Each package
is checked by that producer twice before the current consumer sees its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(binary: Path, root: Path, home: Path) -> dict:
    run = subprocess.run(
        [str(binary), "eval", "--timeout", "30", "--file", str(root / "src/main.ch")],
        cwd=root,
        env={
            **os.environ,
            "CHELIS_STYLE_GATE_DISABLE": "1",
            "CHELIS_REEF_HOME": str(home),
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    return {
        "exit": run.returncode,
        "stdout": run.stdout.strip(),
        "stderr": run.stderr.strip(),
    }


def cache_files(home: Path) -> list[Path]:
    return sorted((home / ".cache/compiled").glob("*.ctx"))


def assert_result(result: dict, family: str, good: bool) -> None:
    if good:
        expected = {
            "reshape": "main = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])",
            "unit": "main = tensor(shape=[3], data=[5.0, 5.0, 5.0])",
            "literal": "main = tensor(shape=[4], data=[7.0, 7.0, 7.0, 7.0])",
        }[family]
        assert result["exit"] == 0 and result["stdout"] == expected, result
    else:
        operation, claim, actual = {
            "reshape": ("reshape", "claimed = 2", "reshape axis 0 = 3"),
            "unit": ("load", "claimed = 1", "b axis 0 = 2"),
            "literal": ("load", "claimed = 4", "x axis 0 = 5"),
        }[family]
        assert result["exit"] != 0, result
        header = f"numeric trap: domain in {operation} at i64"
        assert header in result["stderr"].splitlines(), result
        tokens = re.findall(r"[A-Za-z0-9_.-]+", result["stderr"])
        for record in (claim, actual):
            words = re.findall(r"[A-Za-z0-9_.-]+", record)
            assert any(
                tokens[i : i + len(words)] == words for i in range(len(tokens))
            ), (record, result)


def package(root: Path, version: str, family: str, good: bool) -> None:
    (root / "src").mkdir(parents=True)
    (root / "mylib/src").mkdir(parents=True)
    (root / "reef.toml").write_text(
        f'[package]\nname="app"\nversion="0.1.0"\ncompiler="={version}"\nmodule_prefix="App"\n[dependencies]\nmylib={{path="./mylib"}}\n'
    )
    (root / "mylib/reef.toml").write_text(
        f'[package]\nname="mylib"\nversion="0.1.0"\ncompiler="={version}"\nmodule_prefix="Mylib"\n'
    )
    (root / "reef.lock").write_text(
        f'[package]\nname="app"\nversion="0.1.0"\n[[dependencies]]\nname="mylib"\nversion="0.1.0"\ncompiler="={version}"\narchive_sha256=""\nshell_sha256=""\n[dependencies.source]\nkind="path"\npath="./mylib"\n'
    )
    definition = {
        "reshape": "def f(x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])",
        "unit": "def f(b: tensor[unit, f32]) -> tensor[3, f32] = expand(b, 0i32, 3i64)",
        "literal": "def f(x: tensor[n, f32]) -> tensor[4, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(x, 0i32))",
    }[family]
    values = {
        "reshape": list(range(1, 5 if good else 7)),
        "unit": [5] if good else [5, 6],
        "literal": list(range(1, 5 if good else 6)),
    }[family]
    argument = "to_tensor([" + ", ".join(f"{x}.0f32" for x in values) + "])"
    (root / "mylib/src/claims.ch").write_text(
        f"module Mylib.Claims\nexport (f)\n{definition}\n"
    )
    (root / "src/main.ch").write_text(
        f"module App.Main\nimport Mylib.Claims (f)\ndef main() = f({argument})\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous-head", required=True)
    parser.add_argument("--current-head", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Preserve genuine old producer bytes for the committed rejection fixture",
    )
    args = parser.parse_args()
    previous, current = args.previous.resolve(), args.current.resolve()
    old_hash, new_hash = fingerprint(previous), fingerprint(current)
    assert old_hash != new_hash, (
        "previous/current must be different real producer binaries"
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="chelis-extent-cache-") as directory:
        for family in ("reshape", "unit", "literal"):
            for good in (True, False):
                root = Path(directory) / f"{family}-{good}"
                home = root / "reef-home"
                package(root, args.version, family, good)
                old = execute(previous, root, home)
                old_files = cache_files(home)
                assert len(old_files) == 1, (family, good, old, old_files)
                old_path = old_files[0]
                old_bytes = old_path.read_bytes()
                if args.fixture_dir and family == "reshape" and not good:
                    args.fixture_dir.mkdir(parents=True, exist_ok=True)
                    # Name the file after the magic the producer actually
                    # wrote. A fixed name outlives the format it holds, and a
                    # `context-v18.ctx` carrying V20 bytes is a fixture that
                    # lies about what it proves.
                    magic = old_bytes.split(b"\n", 1)[0].decode()
                    (args.fixture_dir / f"{magic.lower().replace('_', '-')}.ctx").write_bytes(
                        old_bytes
                    )
                    (args.fixture_dir / "producer.json").write_text(
                        json.dumps(
                            {
                                "producer_head": args.previous_head,
                                "producer_binary_sha256": old_hash,
                                "context_sha256": hashlib.sha256(old_bytes).hexdigest(),
                                "magic": magic,
                                "source": (root / "src/main.ch").read_text(),
                                "dependency": (
                                    root / "mylib/src/claims.ch"
                                ).read_text(),
                                "old_result": old,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                old_time = old_path.stat().st_mtime_ns
                old_warm = execute(previous, root, home)
                assert old_warm == old and old_path.stat().st_mtime_ns == old_time, (
                    old,
                    old_warm,
                )
                cold = execute(current, root, home)
                assert_result(cold, family, good)
                new_files = [p for p in cache_files(home) if p != old_path]
                assert len(new_files) == 1, (family, good, new_files)
                new_path = new_files[0]
                current_bytes = new_path.read_bytes()
                assert current_bytes != old_bytes
                # Exercise the inner envelope gate as well as the distinct
                # compiler-identity filename: use genuine old-producer bytes
                # at the exact path the current consumer is about to inspect.
                shutil.copyfile(old_path, new_path)
                repaired = execute(current, root, home)
                assert_result(repaired, family, good)
                assert new_path.read_bytes() == current_bytes, (
                    "old payload was not rejected and rebuilt"
                )
                before_warm = (new_path.stat().st_mtime_ns, fingerprint(new_path))
                warm = execute(current, root, home)
                assert_result(warm, family, good)
                assert before_warm == (
                    new_path.stat().st_mtime_ns,
                    fingerprint(new_path),
                ), "current/current did not hit"
                rows.append(
                    {
                        "family": family,
                        "matching": good,
                        "previous": old,
                        "current_cold": cold,
                        "old_envelope_at_current_path": repaired,
                        "current_warm": warm,
                        "old_payload_sha256": hashlib.sha256(old_bytes).hexdigest(),
                        "current_payload_sha256": fingerprint(new_path),
                    }
                )
                print(
                    f"PASS {family} matching={good}: old/old hit, old/current reject, current/current hit",
                    flush=True,
                )
    args.report.write_text(
        json.dumps(
            {
                "previous_head": args.previous_head,
                "current_head": args.current_head,
                "previous_binary_sha256": old_hash,
                "current_binary_sha256": new_hash,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
