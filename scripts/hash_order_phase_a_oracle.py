#!/usr/bin/env python3
"""Run the complete Phase A hash-order determinism acceptance surface.

Three components were removed with their subject, and what remains is what was
never about it. Phase A ordered the two deferred-shape stores so that the
choice between two candidate `expand` shapes could not depend on hash
iteration order, which is the property chelis#1338 reported as
nondeterministic. `spec/04-type-system.md` section 4.7.2 gives `expand` and
`insert` one result shape each, so there is nothing left to settle: #1338 is
resolved by construction rather than by this oracle.

Gone with the stores: `chelis-cli`'s `hash_order_stability` target, which ran
K=24 fresh processes over programs whose settlement made that choice
(chelis#1277 S2b); the raw-store compile-fail fixture, which proved the stores
exposed no hash iteration API; and the `chelis-types` `hash_order_` unit rows,
which asserted the source ordering of obligations that are no longer recorded
(chelis#1277 S2c).

The four components below never depended on the two-candidate model. The cache
version test pins the serialized checker state, the executable-example and
parity rows exercise `examples/hash_order_determinism.ch` on both lanes, and
`issue_942_inferred_tensor_cast` covers `cast` over a shape-bearing producer.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "cache_format_version_tracks_the_deferred_ledger_removal",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "cli",
        "executable_examples",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--test",
        "issue_942_inferred_tensor_cast",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "parity",
        "parity_hash_order_determinism",
        "--no-fail-fast",
    ),
)


class PhaseAOracleFailure(RuntimeError):
    """One component of the Phase A oracle failed."""


def validate(
    *, runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    commands: Sequence[Sequence[str]] = COMMANDS,
) -> None:
    for command in commands:
        completed = runner(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            rendered = " ".join(command)
            raise PhaseAOracleFailure(
                f"command exited {completed.returncode}: {rendered}"
            )
    print("HASH ORDER PHASE A ORACLE: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except PhaseAOracleFailure as error:
        print(f"HASH ORDER PHASE A ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        raise SystemExit(main())
