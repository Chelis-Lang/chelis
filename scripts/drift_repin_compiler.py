#!/usr/bin/env python3
"""Rewrite a shell's compiler pins to a target chelis version.

Used by the ecosystem drift canary (.github/workflows/ecosystem-drift.yml).

Why this exists: every Chelis shell pins its compiler version as a hard
equality (`compiler = "=X.Y.Z"`), and `chelis reef build` (and `chelis
check` / `chelis test`) reject any manifest whose pin does not exactly
match the running compiler's version (see crates/chelis-reef/src/lib.rs
`validate_manifest`). The canary builds each shell against chelis HEAD,
whose version may be ahead of what the shell currently pins (e.g. a shell
still on `=0.7.26` while HEAD is `0.7.27`). Without rewriting the pin, the
build fails at the manifest gate before exercising any actual language
surface — which is not the drift we want to detect. Rewriting the pin to
HEAD's version asks the real question: "if this shell were bumped to
chelis HEAD, would its source still build and test?"

A shell is not a single manifest: shells carry NESTED reef packages
(consumer smoke projects, examples, `spike/` probes) that each hard-pin
the compiler and would fail the manifest gate the moment a build/check
touches them. So when handed a directory this script repins EVERY
`reef.toml` under it, not just the root. Manifests with no parseable
compiler pin (e.g. workspace-style stubs, negative-test fixtures) are
skipped, not treated as errors — but a run that repins nothing at all
(no `reef.toml` found, or none carrying a pin) fails loudly, because that
silent no-op is exactly the canary-blindness this guards against. Shell CI
also enforces that workflow `CHELIS_TAG` / `CHELIS_VERSION` values agree with
the manifest. Directory mode therefore rewrites those two env keys in every
`.github/workflows/*.yml` and `*.yaml` file as part of the same throwaway pin
bump. Other YAML and other version keys are left untouched.

This is a canary-only, in-CI mutation of a throwaway checkout. It is
never committed and never run against a developer working tree.

Usage:
    # single manifest
    python3 scripts/drift_repin_compiler.py <reef.toml> <target-version>
    # every reef.toml under a shell checkout
    python3 scripts/drift_repin_compiler.py <shell-dir> <target-version>

Example:
    python3 scripts/drift_repin_compiler.py reef.toml 0.7.27
    python3 scripts/drift_repin_compiler.py shell 0.7.27

Exit codes:
    0  pin(s) rewritten (or already matched the target).
    1  nothing to repin: a single manifest with no parseable
       `compiler = "=X.Y.Z"` pin, a missing path, or a directory under
       which no manifest carried a repinnable pin.
    2  wrong number of arguments.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches a top-level `compiler = "=X.Y.Z"` line (quote/space tolerant),
# capturing the surrounding shape so only the version is replaced. The
# `=` equality marker is preserved because reef requires it.
_PIN_RE = re.compile(
    r'(?m)^(?P<key>[ \t]*compiler[ \t]*=[ \t]*"=)\d+\.\d+\.\d+(?P<tail>".*)$'
)

_WORKFLOW_PIN_RE = re.compile(
    r"(?m)^(?P<prefix>[ \t]*(?P<key>CHELIS_TAG|CHELIS_VERSION)[ \t]*:[ \t]*)"
    r"(?P<quote>['\"]?)(?P<version>v?\d+\.\d+\.\d+)(?P=quote)"
    r"(?P<tail>[ \t]*(?:#.*)?)$"
)


def repin(manifest_text: str, target_version: str) -> str:
    """Return `manifest_text` with its compiler pin set to `target_version`.

    Raises ValueError if no `compiler = "=X.Y.Z"` pin is present.
    """
    new_text, count = _PIN_RE.subn(
        lambda m: f'{m.group("key")}{target_version}{m.group("tail")}',
        manifest_text,
    )
    if count == 0:
        raise ValueError(
            'no parseable `compiler = "=X.Y.Z"` pin found in manifest'
        )
    return new_text


def _repin_file(manifest_path: Path, target_version: str) -> bool:
    """Repin one manifest in place. Return True if it had a pin, False if not.

    IO errors propagate; a missing pin is a soft no (False), so a caller
    walking a tree can skip pinless manifests without aborting.
    """
    original = manifest_path.read_text(encoding="utf-8")
    try:
        rewritten = repin(original, target_version)
    except ValueError:
        return False
    manifest_path.write_text(rewritten, encoding="utf-8")
    return True


def repin_workflow(workflow_text: str, target_version: str) -> tuple[str, int]:
    """Rewrite Chelis workflow env pins and return (text, replacements)."""

    def replacement(match: re.Match[str]) -> str:
        version = (
            f"v{target_version}"
            if match.group("key") == "CHELIS_TAG"
            else target_version
        )
        return (
            f'{match.group("prefix")}{match.group("quote")}{version}'
            f'{match.group("quote")}{match.group("tail")}'
        )

    return _WORKFLOW_PIN_RE.subn(replacement, workflow_text)


def _repin_workflows(root: Path, target_version: str) -> tuple[int, int]:
    """Repin workflow files under the conventional GitHub workflow path."""
    workflow_root = root / ".github" / "workflows"
    files = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in workflow_root.glob(pattern)
        if path.is_file()
    )
    rewritten_files = 0
    replacements = 0
    for workflow_path in files:
        original = workflow_path.read_text(encoding="utf-8")
        rewritten, count = repin_workflow(original, target_version)
        if count == 0:
            continue
        workflow_path.write_text(rewritten, encoding="utf-8")
        rewritten_files += 1
        replacements += count
        print(
            f"[drift-repin] {workflow_path}: set {count} Chelis env pin(s) "
            f"to {target_version}"
        )
    return rewritten_files, replacements


def repin_tree(root: Path, target_version: str) -> tuple[int, int]:
    """Repin every `reef.toml` under `root`. Return (repinned, skipped).

    `repinned` counts manifests that carried a pin (and were rewritten);
    `skipped` counts manifests with no parseable pin. Their sum is the
    number of `reef.toml` files found.
    """
    repinned = 0
    skipped = 0
    for manifest_path in sorted(root.rglob("reef.toml")):
        if not manifest_path.is_file():
            continue
        if _repin_file(manifest_path, target_version):
            repinned += 1
            print(
                f"[drift-repin] {manifest_path}: "
                f"compiler pin set to ={target_version}"
            )
        else:
            skipped += 1
            print(f"[drift-repin] {manifest_path}: no compiler pin, skipped")
    _repin_workflows(root, target_version)
    return repinned, skipped


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: drift_repin_compiler.py <reef.toml|shell-dir> "
            "<target-version>",
            file=sys.stderr,
        )
        return 2
    target_path = Path(argv[1])
    target_version = argv[2]

    if target_path.is_dir():
        repinned, skipped = repin_tree(target_path, target_version)
        if repinned == 0:
            found = repinned + skipped
            if found == 0:
                print(
                    f"ERROR: no reef.toml found under {target_path}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"ERROR: {found} reef.toml under {target_path} but none "
                    "carried a repinnable `compiler = \"=X.Y.Z\"` pin",
                    file=sys.stderr,
                )
            return 1
        print(
            f"[drift-repin] {target_path}: repinned {repinned}, "
            f"skipped {skipped} (no pin)"
        )
        return 0

    if not target_path.exists():
        print(
            f"ERROR: {target_path}: no such file or directory",
            file=sys.stderr,
        )
        return 1

    if _repin_file(target_path, target_version):
        print(
            f"[drift-repin] {target_path}: "
            f"compiler pin set to ={target_version}"
        )
        return 0
    print(
        f"ERROR: {target_path}: "
        'no parseable `compiler = "=X.Y.Z"` pin found in manifest',
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
