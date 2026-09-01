#!/usr/bin/env python3
"""Run the Phase B hash-order determinism acceptance surface.

Acceptance is exit 0 with the final line
``HASH ORDER DETERMINISM ORACLE: PASS``.

This script is a command registry, not an analysis. The contract it certifies
lives in three places that each enforce themselves:

- `clippy.toml` bans `std::collections::HashMap`/`HashSet` through Clippy's
  `disallowed_types`, on real HIR, so aliases, glob imports, type-alias
  definition sites, macro expansion, and `include!`-ed generated code are all
  caught by the compiler rather than by pattern matching over source text;
- `scripts/check_configuration_closure.py` proves the configuration space is
  declared and finite, that the registered Clippy matrix covers it, and that
  rustc's own dep-info accounts for every repository `.rs` file;
- the commands below pin the wrapper's behavior, the cache bytes, and the
  existing per-site determinism regressions.

An earlier revision instead reconstructed rustc's compiled-source set from
source text and scanned it for banned spellings. Nine review rounds found
seven distinct ways for real compiled Rust to escape that reconstruction, each
repaired by adding another piece of a Rust lexer, `#[path]` resolver,
module-graph walk, doc-comment desugarer, or macro token-tree inspector. The
compiler is the authority on what the compiler compiles; see
`spec/design/hash_order_determinism.md` C2.3 and its Rationale.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class HashOrderDeterminismFailure(RuntimeError):
    """A Phase B acceptance command failed."""


COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "scripts/check_configuration_closure.py"),
    (sys.executable, "scripts/check_hash_order_phase_b_compile_fail.py"),
    ("cargo", "nextest", "run", "-p", "chelis-unord", "--no-fail-fast"),
    (
        "cargo",
        "test",
        "-p",
        "chelis-unord",
        "--release",
        "--test",
        "serde_contract",
        "debug_is_deterministic_and_does_not_expose_contents",
        "--",
        "--exact",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "hash_order_cache_bytes",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "test",
        "-p",
        "chelis-cli",
        "--test",
        "stdlib_typecheck_cache_oracle",
        "stale_stdlib_byte_mutation_misses_not_stale_hit",
        "--",
        "--exact",
        "--nocapture",
    ),
    (sys.executable, "scripts/hash_order_phase_a_oracle.py"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-backend-c",
        "--test",
        "codegen_determinism",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "rank_poly_tier3",
        "form3_bias_broadcast_c_is_byte_deterministic",
        "--no-fail-fast",
    ),
)


def validate(
    commands: Sequence[Sequence[str]] = COMMANDS,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    for command in commands:
        completed = runner(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            rendered = " ".join(command)
            raise HashOrderDeterminismFailure(
                f"command exited {completed.returncode}: {rendered}"
            )
    print("HASH ORDER DETERMINISM ORACLE: PASS", flush=True)


def main() -> int:
    if sys.argv[1:]:
        print(
            "usage: hash_order_determinism_oracle.py",
            file=sys.stderr,
        )
        return 2
    try:
        validate()
    except (HashOrderDeterminismFailure, OSError, subprocess.SubprocessError) as error:
        print(f"HASH ORDER DETERMINISM ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
