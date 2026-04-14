"""Stamp a new Chelis shell-repo scaffolding tree from an existing one.

Copies a shell-repo directory (e.g. the `nautilus` scaffolding) to a new
destination and rewrites the four scaffolding files that contain the old
shell name:

    reef.toml
    src/core.ch
    README.md
    AGENTS.md

Does NOT touch `.git/`, CI workflows (which reference only `src/core.ch`
and `Chelis-Lang/chelis`, not the shell name), licenses, gitignores,
agent-skill copies, or command wrappers — those are intentionally
identical across shells per the drift-prevention contract in
nautilus/AGENTS.md.

Usage:
    stamp-shell --src ../nautilus --dst ../coral \\
        --old nautilus --new coral \\
        --description "Typed dataframes shell for Chelis" \\
        --phase 3k
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REWRITE_FILES = ("reef.toml", "src/core.ch", "README.md", "AGENTS.md")


def _swap_tokens(text: str, old: str, new: str) -> str:
    """Replace the shell name in both lowercase and Capitalized forms.

    Uppercase is not used in the scaffolding so we don't touch it.
    """
    out = text.replace(old.capitalize(), new.capitalize())
    out = out.replace(old.lower(), new.lower())
    return out


def _rewrite_description(text: str, new_description: str) -> str:
    return re.sub(
        r"(?m)^Numerical methods, statistics, and optimization shell "
        r"for the$",
        new_description.rstrip(".") + " for the",
        text,
    )


def _rewrite_phase(text: str, new_phase: str) -> str:
    return re.sub(r"phase 3j\b", f"phase {new_phase}", text)


def stamp_shell(
    src: Path,
    dst: Path,
    old: str,
    new: str,
    description: str,
    phase: str,
) -> list[Path]:
    """Copy `src` to `dst` and rewrite scaffolding tokens.

    Returns the list of files whose contents were rewritten.
    """
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if not (src / "reef.toml").is_file():
        raise FileNotFoundError(f"source is not a shell repo: {src}")

    def _ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n == ".git"]

    shutil.copytree(src, dst, symlinks=True, ignore=_ignore)

    rewritten: list[Path] = []
    for rel in REWRITE_FILES:
        path = dst / rel
        if not path.is_file():
            continue
        original = path.read_text()
        updated = _swap_tokens(original, old, new)
        updated = _rewrite_description(updated, description)
        updated = _rewrite_phase(updated, phase)
        if updated != original:
            path.write_text(updated)
            rewritten.append(path)

    return rewritten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stamp a new Chelis shell scaffolding tree."
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--old", required=True, help="lowercase old shell name")
    parser.add_argument("--new", required=True, help="lowercase new shell name")
    parser.add_argument("--description", required=True)
    parser.add_argument("--phase", required=True, help="e.g. 3k")
    args = parser.parse_args(argv)

    try:
        rewritten = stamp_shell(
            args.src, args.dst, args.old, args.new, args.description, args.phase
        )
    except (FileExistsError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"stamped {args.dst} from {args.src}")
    for path in rewritten:
        print(f"  rewrote {path.relative_to(args.dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
