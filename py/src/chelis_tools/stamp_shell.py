"""Stamp a new Chelis shell-repo scaffolding tree from an existing one.

Copies a shell-repo directory (e.g. the `nautilus` scaffolding) to a new
destination and rewrites the shell name in the four files that mention
it:

    reef.toml
    src/core.ch
    README.md
    AGENTS.md

Both lowercase (`nautilus`) and capitalized (`Nautilus`) forms are
swapped. Every other file — license, gitignore, CI workflow, agent-skill
copies, command wrappers — is copied verbatim. The `.git/` directory is
excluded.

Usage:
    stamp-shell --src ../nautilus --dst ../coral \\
        --old nautilus --new coral
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REWRITE_FILES = ("reef.toml", "src/core.ch", "README.md", "AGENTS.md")


def _swap_tokens(text: str, old: str, new: str) -> str:
    """Replace the shell name in both lowercase and Capitalized forms."""
    out = text.replace(old.capitalize(), new.capitalize())
    out = out.replace(old.lower(), new.lower())
    return out


def stamp_shell(src: Path, dst: Path, old: str, new: str) -> list[Path]:
    """Copy `src` to `dst` and rewrite shell-name tokens.

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
    args = parser.parse_args(argv)

    try:
        rewritten = stamp_shell(args.src, args.dst, args.old, args.new)
    except (FileExistsError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"stamped {args.dst} from {args.src}")
    for path in rewritten:
        print(f"  rewrote {path.relative_to(args.dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
