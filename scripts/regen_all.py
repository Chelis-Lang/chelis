#!/usr/bin/env python3
"""Regenerate every committed derived artifact that has a writer, in order.

Usage (from the repository root, with the worktree's managed Python):

    .venv/bin/python scripts/regen_all.py               # write tiers 0 and 1
    .venv/bin/python scripts/regen_all.py --check       # report stale legs only
    .venv/bin/python scripts/regen_all.py --tier 0      # the Python-only tier
    .venv/bin/python scripts/regen_all.py --full        # tiers 0, 1, 2 plus the
                                                        # check-only legs
    .venv/bin/python scripts/regen_all.py --full --check

Tiers, in dependency order
--------------------------
0   Python-only writers that need no build: the source-derived rejection issue
    manifest and Rust registry (`generate_rejection_registries.py --write`),
    the embedded conformance skill assets (`regenerate_conformance_assets.py`),
    and the
    opaque-invariants corpus (`tests/corpus/opaque_invariants/generate_corpus.py`).
    The corpus generator has no `--check`; this script regenerates into a
    temporary directory and byte-compares, the same three comparisons
    `crates/chelis-cli/tests/opaque_corpus_gate.rs` makes.
1   The chelis-std bundle (`regenerate_chelis_std_bundle.py --debug`). It needs
    cargo and must follow tier 0 because the rejection registry is compiled
    source.
2   Behind `--full` only. The capacity census, written by running
    `capacity_census_tripwire` with `CHELIS_CAPACITY_CENSUS_WRITE=1`; the
    runtime-representation Phase 0 inventory
    (`runtime_representation_oracle.py --phase 0 --regenerate`, which needs
    clang); and check-only legs for two artifacts that have no writer, the
    generated dtype C header and the wire census JSON. Under
    `--full` the script also prints that the tree-sitter parsers have neither
    a regenerator nor a drift test in this repository.

Write mode stops at the first leg that fails or demands a manual action,
because later tiers consume earlier outputs. `--check` runs every selected leg
and reports all stale ones. A program that cannot be launched (no `cargo` on
PATH, say) is a failed leg in write mode and a stale leg with the launch error
as its reason in check mode; it never escapes as a traceback. Every leg's
writer environment variable (`CHELIS_CAPACITY_CENSUS_WRITE`) is removed from
the child environment before any leg runs and set again only on the write
command that declares it, so an ambient leftover cannot turn a checker into a
writer. The final line is exactly one of:

    REGEN ALL: PASS                              exit 0
    REGEN ALL: STALE (<leg names>)               exit 1  (--check only)
    REGEN ALL: MANUAL ACTION REQUIRED (<names>)  exit 2
    REGEN ALL: FAIL (<leg name>, exit N)         exit 1

Regeneration never blesses growth
---------------------------------
Two tier-2 writers can land content that a human must still review, and this
script refuses to call that a pass:

* The capacity census writer gives every new row the citation `"TODO"`. After
  the write, the script scans `spec/design/capacity_census.json` and exits 2
  with the classification instruction if any such row exists.
* The runtime-representation regeneration rewrites the inventory JSON, but the
  reviewed `FREEZE_SHA256` literal in `scripts/runtime_representation_oracle.py`
  is a B1 freeze move. After the write, the script compares the inventory's
  `freeze_sha256` to that literal and exits 2 naming the move if they differ.

What this script never writes
-----------------------------
* `FROZEN_ATOM_DIGESTS` and `FROZEN_REGION_DIGESTS` in
  `scripts/dtype_phase4b_oracle.py`;
* `FREEZE_SHA256` in `scripts/runtime_representation_oracle.py`;
* the `BASELINE` table in `crates/chelis-cli/tests/loud_unsupported_tripwire.rs`;
* `scripts/runtime_extent_oracle_baseline.json` and
  `scripts/runtime_extent_oracle_baseline_phase_b.json`;
* `docs/copy_drop_fixture_fitness_baseline.json`;
* `scripts/test_timing_baseline.json` (regenerated from CI telemetry only);
* `spec/design/capacity_census_bindings.json` (stable reviewed authority rows,
  not execution-derived graph hashes);
* `spec/design/capacity_census_wire.json` (no writer exists);
* `crates/chelis-runtime/include/chelis_runtime_dtype.h` (no writer exists);
* the tree-sitter parsers under `grammars/`.

This is an ordinary uv-managed script: invoke it through `.venv/bin/python`
(or let `scripts/gate.py --fast` do so). It has no bootstrap-free carve-out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_MANUAL = 2

DEFAULT_TIERS: tuple[int, ...] = (0, 1)
FULL_TIERS: tuple[int, ...] = (0, 1, 2)

CENSUS_JSON = "spec/design/capacity_census.json"
CENSUS_WRITE_ENV = "CHELIS_CAPACITY_CENSUS_WRITE"
CENSUS_TODO_CITATION = "TODO"
RUNTIME_REPRESENTATION_ORACLE = "scripts/runtime_representation_oracle.py"
RUNTIME_REPRESENTATION_INVENTORY = (
    "spec/design/runtime_representation_phase0_inventory.json"
)
FREEZE_SHA_LITERAL = re.compile(
    r'^FREEZE_SHA256\s*=\s*"(?P<sha>[0-9a-f]{64})"', re.MULTILINE
)
OPAQUE_CORPUS_DIR = "tests/corpus/opaque_invariants"
OPAQUE_CORPUS_GENERATOR = f"{OPAQUE_CORPUS_DIR}/generate_corpus.py"

TREE_SITTER_NOTE = (
    "grammars/tree-sitter-chelis-surf/src and grammars/tree-sitter-chelis-deep/src "
    "are `tree-sitter generate` output with no regenerator and no drift test in "
    "this repository; regen_all cannot verify them."
)
DTYPE_HEADER_MANUAL = (
    "no writer exists for crates/chelis-runtime/include/chelis_runtime_dtype.h; "
    "hand-copy the output of render_runtime_dtype_c_header() "
    "(crates/chelis-runtime/src/dtype_header.rs) into that header, then rerun "
    "this leg."
)
CENSUS_WIRE_MANUAL = (
    "no --write seam exists for spec/design/capacity_census_wire.json; "
    "edit it by hand from scripts/capacity_census_typed.py output, then "
    "rerun this leg."
)
CENSUS_TODO_INSTRUCTION = (
    "classify each row through exactly one final authority (rules 1-3 of the "
    "capacity_census_tripwire footer: structurally nonnumeric, tagged transport, "
    "or numeric operation with its exact [05-OP-N] atom), rerun tier 0 so the "
    "rejection registry picks up any new atom, then rerun this leg."
)


@dataclass(frozen=True)
class RegenLeg:
    """One regeneration leg: what it writes, how to write it, how to check it.

    `write_argv` is None for an artifact that has no writer (check-only leg).
    `check_argv` is None when the check is a custom function named by
    `custom_check`, or when the leg is only a printed `note`. `env` is applied
    to the write command only; `--check` never sets it. `after_write` names a
    hook run after a successful write that may demand a manual action.
    """

    name: str
    tier: int
    write_argv: tuple[str, ...] | None
    check_argv: tuple[str, ...] | None
    writes: tuple[str, ...]
    needs: str
    env: tuple[tuple[str, str], ...] = ()
    manual_after: str | None = None
    after_write: str | None = None
    custom_check: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class LegResult:
    leg: RegenLeg
    status: str  # ok | stale | failed | manual | note
    returncode: int = 0
    detail: str | None = None


def regen_legs(python: str) -> tuple[RegenLeg, ...]:
    """The frozen leg manifest in execution order (tiers ascending)."""
    return (
        RegenLeg(
            name="rejection-registry",
            tier=0,
            write_argv=(python, "scripts/generate_rejection_registries.py", "--write"),
            check_argv=(python, "scripts/generate_rejection_registries.py", "--check"),
            writes=(
                "spec/design/loud_unsupported_issue_manifest.json",
                "crates/chelis-types/src/rejection_registry_generated.rs",
            ),
            needs="python",
        ),
        RegenLeg(
            name="conformance-assets",
            tier=0,
            write_argv=(python, "scripts/regenerate_conformance_assets.py"),
            check_argv=(python, "scripts/regenerate_conformance_assets.py", "--check"),
            writes=("crates/chelis-conformance/assets/skills/",),
            needs="python",
        ),
        RegenLeg(
            name="opaque-corpus",
            tier=0,
            write_argv=(python, OPAQUE_CORPUS_GENERATOR),
            check_argv=None,
            writes=(
                f"{OPAQUE_CORPUS_DIR}/manifest.json",
                f"{OPAQUE_CORPUS_DIR}/programs/",
            ),
            needs="python",
            custom_check="opaque-corpus",
        ),
        RegenLeg(
            name="std-bundle",
            tier=1,
            write_argv=(python, "scripts/regenerate_chelis_std_bundle.py", "--debug"),
            check_argv=(
                python,
                "scripts/regenerate_chelis_std_bundle.py",
                "--debug",
                "--check",
            ),
            writes=(
                "packages/chelis-std/dist/",
                "packages/chelis-std/reef.lock",
                "crates/chelis-std-bundle/dist/",
            ),
            needs="cargo",
        ),
        RegenLeg(
            name="capacity-census",
            tier=2,
            write_argv=(
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
            ),
            check_argv=(
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
            ),
            writes=(CENSUS_JSON,),
            needs="cargo",
            env=((CENSUS_WRITE_ENV, "1"),),
            after_write="census-todo-rows",
        ),
        RegenLeg(
            name="runtime-representation",
            tier=2,
            write_argv=(
                python,
                RUNTIME_REPRESENTATION_ORACLE,
                "--phase",
                "0",
                "--regenerate",
            ),
            check_argv=(python, RUNTIME_REPRESENTATION_ORACLE, "--phase", "0"),
            writes=(RUNTIME_REPRESENTATION_INVENTORY,),
            needs="cargo+clang",
            after_write="freeze-sha",
        ),
        RegenLeg(
            name="dtype-c-header",
            tier=2,
            write_argv=None,
            check_argv=(
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-runtime",
                "--test",
                "runtime_dtype_generated_header",
            ),
            writes=("crates/chelis-runtime/include/chelis_runtime_dtype.h",),
            needs="cargo",
            manual_after=DTYPE_HEADER_MANUAL,
        ),
        RegenLeg(
            name="capacity-census-wire",
            tier=2,
            write_argv=None,
            check_argv=(
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-compiler-api",
                "--test",
                "capacity_census_wire",
            ),
            writes=("spec/design/capacity_census_wire.json",),
            needs="cargo",
            manual_after=CENSUS_WIRE_MANUAL,
        ),
        RegenLeg(
            name="tree-sitter",
            tier=2,
            write_argv=None,
            check_argv=None,
            writes=(
                "grammars/tree-sitter-chelis-surf/src/",
                "grammars/tree-sitter-chelis-deep/src/",
            ),
            needs="none",
            note=TREE_SITTER_NOTE,
        ),
    )


def select_legs(
    legs: tuple[RegenLeg, ...], tiers: tuple[int, ...]
) -> list[RegenLeg]:
    """Legs whose tier is selected, in manifest order (tiers ascend)."""
    selected = [leg for leg in legs if leg.tier in tiers]
    assert selected == sorted(selected, key=lambda leg: leg.tier)
    return selected


# --- post-write hooks: a manual-action message, or None ---------------------


def census_todo_rows(census_path: Path) -> list[dict]:
    """Rows the census writer landed with the unclassified `"TODO"` citation."""
    document = json.loads(census_path.read_text(encoding="utf-8"))
    rows = document.get("rows", []) if isinstance(document, dict) else []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("citation") == CENSUS_TODO_CITATION
    ]


def after_census_write(repo_root: Path) -> str | None:
    rows = census_todo_rows(repo_root / CENSUS_JSON)
    if not rows:
        return None
    lines = [
        f"{len(rows)} capacity census row(s) landed with citation "
        f'"{CENSUS_TODO_CITATION}" in {CENSUS_JSON}:'
    ]
    for row in rows:
        lines.append(f"  {row.get('kind', '?')}: {row.get('id', '?')}")
    lines.append(CENSUS_TODO_INSTRUCTION)
    return "\n".join(lines)


def reviewed_freeze_sha(oracle_source: str) -> str | None:
    match = FREEZE_SHA_LITERAL.search(oracle_source)
    return None if match is None else match.group("sha")


def after_runtime_representation_write(repo_root: Path) -> str | None:
    inventory_path = repo_root / RUNTIME_REPRESENTATION_INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    written = inventory.get("freeze_sha256") if isinstance(inventory, dict) else None
    oracle_path = repo_root / RUNTIME_REPRESENTATION_ORACLE
    reviewed = reviewed_freeze_sha(oracle_path.read_text(encoding="utf-8"))
    if reviewed is None:
        return (
            f"could not find the FREEZE_SHA256 literal in "
            f"{RUNTIME_REPRESENTATION_ORACLE}; inspect the file by hand."
        )
    if written == reviewed:
        return None
    return (
        f"the regenerated {RUNTIME_REPRESENTATION_INVENTORY} hashes to "
        f"{written}, but {RUNTIME_REPRESENTATION_ORACLE} pins FREEZE_SHA256 = "
        f'"{reviewed}". Move the literal by hand under the B1 freeze review '
        "(spec/design/runtime_representation.md); regen_all never edits it."
    )


AFTER_WRITE_HOOKS: dict[str, Callable[[Path], str | None]] = {
    "census-todo-rows": after_census_write,
    "freeze-sha": after_runtime_representation_write,
}


# --- custom checks: a list of staleness reasons, empty when current -----------


def _program_files(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        return {}
    return {
        entry.name: entry.read_bytes()
        for entry in sorted(directory.iterdir())
        if entry.is_file()
    }


def compare_corpus_trees(committed: Path, regenerated: Path) -> list[str]:
    """The three comparisons `opaque_corpus_gate.rs` makes, as reasons."""
    reasons: list[str] = []
    committed_manifest = committed / "manifest.json"
    regenerated_manifest = regenerated / "manifest.json"
    if not regenerated_manifest.is_file():
        reasons.append("the generator produced no manifest.json")
    elif not committed_manifest.is_file():
        reasons.append("committed manifest.json is missing")
    elif committed_manifest.read_bytes() != regenerated_manifest.read_bytes():
        reasons.append("committed manifest.json is stale vs generate_corpus.py")
    committed_programs = _program_files(committed / "programs")
    regenerated_programs = _program_files(regenerated / "programs")
    for name, content in committed_programs.items():
        if name not in regenerated_programs:
            reasons.append(f"committed program {name!r} has no regenerated twin")
        elif content != regenerated_programs[name]:
            reasons.append(f"committed program {name!r} is stale vs generate_corpus.py")
    for name in regenerated_programs:
        if name not in committed_programs:
            reasons.append(f"regenerated program {name!r} is missing from programs/")
    return reasons


def check_opaque_corpus(
    repo_root: Path, python: str, runner, environ: dict[str, str]
) -> list[str]:
    corpus = repo_root / OPAQUE_CORPUS_DIR
    with tempfile.TemporaryDirectory(prefix="chelis-regen-corpus-") as scratch:
        completed = launch(
            runner,
            [python, OPAQUE_CORPUS_GENERATOR, "--out-dir", scratch],
            cwd=repo_root,
            env=environ,
        )
        if _launch_error(completed) is not None:
            return [_launch_error(completed)]
        if completed.returncode != 0:
            return [f"generate_corpus.py exited {completed.returncode}"]
        return compare_corpus_trees(corpus, Path(scratch))


CUSTOM_CHECKS: dict[str, Callable[[Path, str, object, dict[str, str]], list[str]]] = {
    "opaque-corpus": check_opaque_corpus,
}


# --- the runner --------------------------------------------------------------


LAUNCH_FAILURE_EXIT = 127


def leg_env_keys(legs: tuple[RegenLeg, ...]) -> frozenset[str]:
    """Every writer seam any leg declares, selected or not."""
    return frozenset(name for leg in legs for name, _value in leg.env)


def scrub_environment(environ: dict[str, str], keys: frozenset[str]) -> dict[str, str]:
    """The child environment with every leg writer seam removed. `--check`
    children get exactly this; a write command gets it plus its own `env`."""
    return {name: value for name, value in environ.items() if name not in keys}


def launch(runner, argv: list[str], *, cwd: Path, env: dict[str, str]):
    """Run one leg command; a launch failure (missing program, permission)
    becomes a CompletedProcess with exit 127 and the error text, so every
    path still reaches the final `REGEN ALL:` line."""
    try:
        return runner(argv, cwd=cwd, env=env, check=False)
    except OSError as exc:
        completed = subprocess.CompletedProcess(argv, LAUNCH_FAILURE_EXIT)
        completed.launch_error = f"could not launch {argv[0]}: {exc}"
        return completed


def _launch_error(completed) -> str | None:
    return getattr(completed, "launch_error", None)


def _render(leg: RegenLeg, argv: tuple[str, ...], with_env: bool) -> str:
    prefix = ""
    if with_env and leg.env:
        prefix = " ".join(f"{name}={value}" for name, value in leg.env) + " "
    return prefix + shlex.join(argv)


def run_leg(
    leg: RegenLeg,
    *,
    check: bool,
    repo_root: Path,
    python: str,
    runner,
    environ: dict[str, str],
    out,
    position: str,
) -> LegResult:
    header = f"[tier {leg.tier} {position}] {leg.name} ({leg.needs})"
    if leg.note is not None:
        print(f"{header}: NOTE {leg.note}", file=out, flush=True)
        return LegResult(leg, "note")

    if check:
        if leg.custom_check is not None:
            print(f"{header}: custom check {leg.custom_check}", file=out, flush=True)
            reasons = CUSTOM_CHECKS[leg.custom_check](repo_root, python, runner, environ)
            for reason in reasons:
                print(f"  stale: {reason}", file=out, flush=True)
            if reasons:
                return LegResult(leg, "stale", 1, "; ".join(reasons))
            return LegResult(leg, "ok")
        assert leg.check_argv is not None
        print(f"{header}: {_render(leg, leg.check_argv, False)}", file=out, flush=True)
        completed = launch(runner, list(leg.check_argv), cwd=repo_root, env=environ)
        if _launch_error(completed) is not None:
            print(f"  stale: {_launch_error(completed)}", file=out, flush=True)
            return LegResult(leg, "stale", completed.returncode, _launch_error(completed))
        if completed.returncode != 0:
            detail = leg.manual_after
            if detail is not None:
                print(f"  manual: {detail}", file=out, flush=True)
            return LegResult(leg, "stale", completed.returncode, detail)
        return LegResult(leg, "ok")

    if leg.write_argv is None:
        # Check-only leg in write mode: the artifact has no writer, so a
        # failing check is a manual action, never something to fix here.
        assert leg.check_argv is not None
        print(
            f"{header}: (no writer) {_render(leg, leg.check_argv, False)}",
            file=out,
            flush=True,
        )
        completed = launch(runner, list(leg.check_argv), cwd=repo_root, env=environ)
        if _launch_error(completed) is not None:
            print(f"  failed: {_launch_error(completed)}", file=out, flush=True)
            return LegResult(leg, "failed", completed.returncode, _launch_error(completed))
        if completed.returncode != 0:
            print(f"  manual: {leg.manual_after}", file=out, flush=True)
            return LegResult(leg, "manual", completed.returncode, leg.manual_after)
        return LegResult(leg, "ok")

    print(f"{header}: {_render(leg, leg.write_argv, True)}", file=out, flush=True)
    # `environ` arrives scrubbed of every leg's writer seam; only this leg's
    # own declaration is set, on this one command.
    leg_environ = dict(environ)
    leg_environ.update(dict(leg.env))
    completed = launch(runner, list(leg.write_argv), cwd=repo_root, env=leg_environ)
    if _launch_error(completed) is not None:
        print(f"  failed: {_launch_error(completed)}", file=out, flush=True)
        return LegResult(leg, "failed", completed.returncode, _launch_error(completed))
    if completed.returncode != 0:
        return LegResult(leg, "failed", completed.returncode)
    if leg.after_write is not None:
        manual = AFTER_WRITE_HOOKS[leg.after_write](repo_root)
        if manual is not None:
            print(f"  manual: {manual}", file=out, flush=True)
            return LegResult(leg, "manual", 0, manual)
    if leg.manual_after is not None:
        print(f"  note: {leg.manual_after}", file=out, flush=True)
    print(f"  owns: {', '.join(leg.writes)}", file=out, flush=True)
    return LegResult(leg, "ok")


def verdict(results: list[LegResult]) -> tuple[str, int]:
    failed = [r for r in results if r.status == "failed"]
    if failed:
        first = failed[0]
        return (
            f"REGEN ALL: FAIL ({first.leg.name}, exit {first.returncode})",
            EXIT_FAIL,
        )
    manual = [r.leg.name for r in results if r.status == "manual"]
    if manual:
        return f"REGEN ALL: MANUAL ACTION REQUIRED ({', '.join(manual)})", EXIT_MANUAL
    stale = [r.leg.name for r in results if r.status == "stale"]
    if stale:
        return f"REGEN ALL: STALE ({', '.join(stale)})", EXIT_FAIL
    return "REGEN ALL: PASS", EXIT_PASS


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the committed derived artifacts in dependency order, "
            "or check them with --check."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run every selected leg's checker and report all stale legs; write nothing",
    )
    parser.add_argument(
        "--tier",
        action="append",
        type=int,
        choices=FULL_TIERS,
        metavar="N",
        help="select a tier (repeatable); default 0 and 1, or 0, 1, 2 with --full",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "also run tier 2 (the censuses, the runtime-representation inventory, "
            "the binding graph writer, and the check-only legs)"
        ),
    )
    args = parser.parse_args(argv)
    if args.tier is None:
        args.tiers = FULL_TIERS if args.full else DEFAULT_TIERS
    else:
        args.tiers = tuple(sorted(set(args.tier)))
    if 2 in args.tiers and not args.full:
        parser.error("--tier 2 requires --full")
    return args


def main(
    argv: list[str],
    *,
    repo_root: Path = REPO_ROOT,
    python: str | None = None,
    runner=None,
    environ: dict[str, str] | None = None,
    out=None,
) -> int:
    args = parse_args(argv)
    stream = sys.stdout if out is None else out
    interpreter = sys.executable if python is None else python
    run = subprocess.run if runner is None else runner
    all_legs = regen_legs(interpreter)
    # Scrub every writer seam any leg declares, whether or not that leg is
    # selected: an ambient CHELIS_CAPACITY_CENSUS_WRITE=1 must not make the
    # census checker write and then compare against what it just wrote.
    child_environ = scrub_environment(
        dict(os.environ if environ is None else environ), leg_env_keys(all_legs)
    )

    legs = select_legs(all_legs, args.tiers)
    mode = "check" if args.check else "write"
    print(
        f"regen_all: {mode} mode, tiers {', '.join(str(t) for t in args.tiers)}, "
        f"{len(legs)} leg(s)",
        file=stream,
        flush=True,
    )
    results: list[LegResult] = []
    total = len(legs)
    for index, leg in enumerate(legs, start=1):
        result = run_leg(
            leg,
            check=args.check,
            repo_root=repo_root,
            python=interpreter,
            runner=run,
            environ=child_environ,
            out=stream,
            position=f"{index}/{total}",
        )
        results.append(result)
        if not args.check and result.status in ("failed", "manual"):
            remaining = total - index
            if remaining:
                print(
                    f"regen_all: stopping; {remaining} later leg(s) not run because "
                    "they consume this leg's output",
                    file=stream,
                    flush=True,
                )
            break
    line, code = verdict(results)
    print(line, file=stream, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
