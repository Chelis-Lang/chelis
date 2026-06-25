#!/usr/bin/env python3
"""Fail fast, with a clear diagnostic, when the Hull Conformance nightly's
live fresh campaign cannot possibly pass because the live chelis binary
version does not match the compiler pin baked into the checked-out Hull repo.

Why this exists (chelis#341 / #335): the nightly's "Fresh campaign (live Hull
reference)" step builds the chelis binary at HEAD and runs Hull's own
`run_differential_suite.py` against it. Hull is a Chelis SHELL: its `reef.toml`
pins the compiler as a hard equality (`compiler = "=X.Y.Z"`), and
`chelis reef build` REJECTS any manifest whose pin is not byte-equal to the
running compiler's version. The Hull commit the nightly checks out is pinned
by the chelis-side conformance manifest (`tests/conformance/hull/manifest.json`
-> `hull_commit`), and that pin is bumped only during a release cascade.

When chelis is released ahead of a Hull cascade -- e.g. chelis is 0.10.0 but
the pinned Hull commit is still the 0.7.27-era one whose deps pin `=0.8.0` --
EVERY generated program is rejected at `reef build` with

    error: package.compiler must be `=0.8.0` in <ctx>

and each campaign shard dies with `missing report (no report produced, rc=2)`.
Before this check, that surfaced as four opaque shard failures with no
indication of the real cause (a cross-repo cascade lag, not a code
regression). This script turns that into a single, actionable line up front.

It does NOT paper over the failure: when the versions are skewed it STILL
exits nonzero (the nightly stays red, because the fresh campaign genuinely
cannot run until Hull is cascaded). It only replaces an opaque multi-shard
crash with a diagnosis. The per-PR FROZEN conformance gate (conformance.yml)
is unaffected -- it replays frozen strings and never runs a Hull `reef build`,
so it has no version-match requirement.

The fix when this fires is the cross-repo cascade: cascade Hull to the new
chelis version (its reef.toml + dep pins), then bump `hull_commit` in
`tests/conformance/hull/manifest.json` to the cascaded Hull commit in
lockstep. See the project plan's shell-cascade procedure.

Usage:

    python3 scripts/ci_check_hull_pin_skew.py \
        --chelis-bin <path/to/chelis> \
        --hull-reef-toml <path/to/hull/reef.toml>

Exit codes:
    0  the live binary version matches the Hull compiler pin (campaign can run)
    1  SKEW: versions differ -> the fresh campaign cannot pass; cascade Hull
    2  usage / parse error (could not read a version or a pin)

Per repo policy this is Python, not a shell script.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# `compiler = "=X.Y.Z"` line in a shell reef.toml, quote/space tolerant.
# Mirrors scripts/drift_repin_compiler.py's _PIN_RE so the two stay in
# agreement about what a compiler pin looks like.
_PIN_RE = re.compile(
    r'(?m)^[ \t]*compiler[ \t]*=[ \t]*"=(?P<version>\d+\.\d+\.\d+)"'
)

# `chelis --version` prints `chelis X.Y.Z`.
_VERSION_RE = re.compile(r"\bchelis[ \t]+(?P<version>\d+\.\d+\.\d+)\b")


def parse_hull_compiler_pin(reef_toml_text: str) -> str:
    """Return the `=X.Y.Z` compiler version Hull's reef.toml pins.

    Raises ValueError if no parseable pin is present."""
    m = _PIN_RE.search(reef_toml_text)
    if not m:
        raise ValueError(
            'no parseable `compiler = "=X.Y.Z"` pin found in Hull reef.toml'
        )
    return m.group("version")


def parse_chelis_version(version_output: str) -> str:
    """Return the X.Y.Z version from `chelis --version` output.

    Raises ValueError if no version token is present."""
    m = _VERSION_RE.search(version_output)
    if not m:
        raise ValueError(
            f"could not parse a chelis version from: {version_output!r}"
        )
    return m.group("version")


def query_chelis_version(chelis_bin: str, *, run=subprocess.run) -> str:
    """Run `<chelis_bin> --version` and parse the version out of it."""
    proc = run(
        [chelis_bin, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"`{chelis_bin} --version` exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()!r}"
        )
    return parse_chelis_version(proc.stdout or proc.stderr or "")


def check_skew(live_version: str, hull_pin: str) -> int:
    """Return 0 if the live binary matches the Hull pin, else print a clear
    diagnostic and return 1. `chelis reef build` requires byte-exact
    equality, so this is a string compare, not semver-range matching."""
    if live_version == hull_pin:
        print(
            f"ci_check_hull_pin_skew: OK -- live chelis {live_version} "
            f"matches the checked-out Hull compiler pin =={hull_pin}; "
            f"the fresh campaign can run."
        )
        return 0

    print(
        "ci_check_hull_pin_skew: SKEW -- the Hull Conformance fresh campaign "
        "cannot pass.\n"
        f"  live chelis binary : {live_version}\n"
        f"  Hull reef.toml pin : ={hull_pin}\n"
        "\n"
        "`chelis reef build` rejects any manifest whose `compiler = \"=X.Y.Z\"` "
        "pin is not byte-equal to the running compiler, so every generated "
        "program in the fresh campaign would be rejected (you would otherwise "
        "see `package.compiler must be `=X.Y.Z`` -- from this top-level pin or "
        "a transitively-pinned Hull dependency -- and rc=2 on every shard).\n"
        "\n"
        "This is a CROSS-REPO CASCADE LAG, not a code regression: chelis was "
        "released ahead of a Hull cascade. The fix is to cascade Hull to "
        f"chelis {live_version} (its reef.toml + dep pins), then bump "
        "`hull_commit` in tests/conformance/hull/manifest.json to the "
        "cascaded Hull commit in lockstep. See chelis#341 / #335.",
        file=sys.stderr,
    )
    return 1


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail fast when the live chelis version cannot satisfy the "
            "checked-out Hull compiler pin (chelis#341/#335)."
        )
    )
    parser.add_argument(
        "--chelis-bin",
        required=True,
        help="path to the built chelis binary (its --version is the live version).",
    )
    parser.add_argument(
        "--hull-reef-toml",
        required=True,
        help="path to the checked-out Hull repo's reef.toml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        live_version = query_chelis_version(args.chelis_bin)
        hull_pin = parse_hull_compiler_pin(_read_text(args.hull_reef_toml))
    except (ValueError, OSError) as exc:
        print(f"ci_check_hull_pin_skew: {exc}", file=sys.stderr)
        return 2
    return check_skew(live_version, hull_pin)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
