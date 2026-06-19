#!/usr/bin/env python3
"""Rewrite a shell's `reef.toml` compiler pin to a target chelis version.

Used by the ecosystem drift canary (.github/workflows/ecosystem-drift.yml).

Why this exists: every Chelis shell pins its compiler version as a hard
equality (`compiler = "=X.Y.Z"`), and `chelis reef build` rejects any
manifest whose pin does not exactly match the running compiler's version
(see crates/chelis-reef/src/lib.rs `validate_manifest`). The canary builds
each shell against chelis HEAD, whose version may be ahead of what the
shell currently pins (e.g. a shell still on `=0.7.26` while HEAD is
`0.7.27`). Without rewriting the pin, the build fails at the manifest
gate before exercising any actual language surface — which is not the
drift we want to detect. Rewriting the pin to HEAD's version asks the
real question: "if this shell were bumped to chelis HEAD, would its
source still build and test?"

This is a canary-only, in-CI mutation of a throwaway checkout. It is
never committed and never run against a developer working tree.

Usage:
    python3 scripts/drift_repin_compiler.py <reef.toml> <target-version>

Example:
    python3 scripts/drift_repin_compiler.py reef.toml 0.7.27

Exit codes:
    0  pin rewritten (or already matched the target).
    1  the manifest had no parseable `compiler = "=X.Y.Z"` pin.
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


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: drift_repin_compiler.py <reef.toml> <target-version>",
            file=sys.stderr,
        )
        return 2
    manifest_path = Path(argv[1])
    target_version = argv[2]

    original = manifest_path.read_text(encoding="utf-8")
    try:
        rewritten = repin(original, target_version)
    except ValueError as exc:
        print(f"ERROR: {manifest_path}: {exc}", file=sys.stderr)
        return 1

    manifest_path.write_text(rewritten, encoding="utf-8")
    print(
        f"[drift-repin] {manifest_path}: compiler pin set to ={target_version}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
