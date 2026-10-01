#!/usr/bin/env python3
"""Kernel-check Chelis's production permutation gate with Vermilion/Lean."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
PIN = "696756d6b9bbaedd61d6cd743ec3165f2b639224"


def run(*args: str | Path, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def exact_item(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise ValueError(f"unclosed Verus item: {marker}")


def main() -> None:
    override = os.environ.get("CHELIS_VERMILION_HOME")
    home = Path(override) if override else Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "chelis/vermilion" / PIN
    if not (home / ".git").is_dir():
        if override:
            raise SystemExit(f"CHELIS_VERMILION_HOME is not a Git checkout: {home}")
        home.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/ilyasergey/vermilion.git", home, cwd=REPO)
        run("git", "checkout", "--detach", PIN, cwd=home)
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=home, capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual != PIN:
        raise SystemExit(f"Vermilion checkout is {actual}, expected {PIN}")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.elan/bin'}{os.pathsep}{env['PATH']}"
    env["RUSTUP_NO_SELF_UPDATE"] = "1"
    if not (home / "target/debug/vrml_check").is_file() or not (
        home / ".verus-checkout/source/target-verus/release/rust_verify"
    ).is_file():
        run("./scripts/build.sh", cwd=home, env=env)

    source = (REPO / "crates/chelis-axis-core/src/verified.rs").read_text()
    spec = exact_item(source, "    pub open spec fn valid_permutation")
    function = exact_item(source, "    pub fn is_permutation")
    if source.count(spec) != 1 or source.count(function) != 1:
        raise SystemExit("cannot extract unique production permutation items")
    case_dir = home / "examples/chelis-permutation"
    case_dir.mkdir(exist_ok=True)
    comparison = case_dir / "axis_perm.rs"
    comparison.write_text("use vstd::prelude::*;\n\nverus! {\n" + spec + "\n\n" + function + "\n}\n")
    print(f"Chelis verified source SHA-256: {hashlib.sha256(source.encode()).hexdigest()}")
    print(f"Vermilion permutation input SHA-256: {hashlib.sha256(comparison.read_bytes()).hexdigest()}")
    sys.stdout.flush()

    with (case_dir / "initial-run.log").open("w") as log:
        first = subprocess.run(
            ["./scripts/run_example.sh", str(case_dir), "axis_perm.rs", "--per-file"],
            cwd=home,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if first.returncode not in (0, 1):
        print((case_dir / "initial-run.log").read_text(), file=sys.stderr)
        raise SystemExit(first.returncode)
    run(
        sys.executable,
        REPO / "crates/chelis-axis-core/proofs/axis_vermilion_proofs.py",
        case_dir / "proofs/axis_perm.lean",
        cwd=REPO,
        env=env,
    )
    run(
        "./scripts/run_example.sh", case_dir, "axis_perm.rs", "--per-file",
        "--manual-proofs", "--lib", "Vermilion", cwd=home, env=env,
    )

    negative_dir = home / "examples/chelis-permutation-negative"
    negative_dir.mkdir(exist_ok=True)
    original = "valid == valid_permutation(axes@, rank)"
    comparison_source = comparison.read_text()
    if comparison_source.count(original) != 1:
        raise SystemExit("cannot locate the exact permutation postcondition")
    (negative_dir / "axis_perm_false.rs").write_text(
        comparison_source.replace(original, "valid == false")
    )
    negative = subprocess.run(
        [
            "./scripts/run_example.sh", str(negative_dir), "axis_perm_false.rs",
            "--per-file", "--expect-failure",
        ],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
    )
    if negative.returncode != 0 or "postcondition not satisfied" not in negative.stderr:
        print(negative.stdout, negative.stderr, file=sys.stderr)
        raise SystemExit("Vermilion did not reject the false permutation postcondition")
    print("Vermilion rejected the false permutation postcondition")


if __name__ == "__main__":
    main()
