#!/usr/bin/env python3
"""Bump the workspace version and every coupled compiler pin in lockstep.

Background: when `workspace.package.version` in the root `Cargo.toml`
changes, three categories of files must change with it:

1. Test fixtures with hardcoded `compiler = "=X.Y.Z"` strings. These are
   already auto-synced via `chelis_compiler_api::COMPILER_VERSION`
   (which uses `env!("CARGO_PKG_VERSION")` at compile time). This script
   does NOT touch them — the auto-sync handles the work.

2. Real `.toml` files that ship in the repo and must hand-pin a compiler
   version. The integration tests installs `chelis-std` from
   `packages/chelis-std/` and compares its embedded `package.compiler`
   pin against the running binary's version, so a stale pin fails CI.

3. The prebuilt `packages/chelis-std/dist/*.chb` and `*.tar.zst` artifacts.
   These are produced by `chelis reef build` and embed the bumped pin.

This script is the single, scriptable entry point for the release bump.
The corresponding tripwire test
(`crates/chelis-cli/tests/compiler_pin_tripwire.rs`) fails loudly when
category (2) drifts from category (1), pointing future operators at this
script.

Usage:
    python3 scripts/bump_compiler_pins.py 0.3.2

The script is safe to run from any cwd: paths are resolved relative to
the script's own location.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real `.toml` files whose `package.compiler` pin must equal `=<new-version>`.
# Keep in sync with the same list in `compiler_pin_tripwire.rs`.
PINNED_REAL_TOML_FILES: list[Path] = [
    REPO_ROOT / "packages/chelis-std/reef.toml",
    REPO_ROOT / "crates/chelis-cli/tests/fixtures/pseudo_nautilus/reef.toml",
    REPO_ROOT / "crates/chelis-cli/tests/fixtures/release_pipe_stage/reef.toml",
    REPO_ROOT / "examples/illustrative/phase3g_text_pipeline/reef.toml",
]

WORKSPACE_CARGO_TOML = REPO_ROOT / "Cargo.toml"

# Files whose `chelis-std` dist artifacts get rebuilt at the end. The dir
# itself is enumerated; the names are checked after rebuild.
CHELIS_STD_DIR = REPO_ROOT / "packages/chelis-std"
CHELIS_STD_DIST = CHELIS_STD_DIR / "dist"


SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$")


@dataclass
class FileChange:
    path: Path
    before: str
    after: str

    def render(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"  {rel}: {self.before!r} -> {self.after!r}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bump workspace.package.version and every real .toml `package.compiler` "
            "pin, then rebuild packages/chelis-std/dist/."
        ),
    )
    p.add_argument(
        "version",
        help="New workspace version, e.g. 0.3.2 (semver)",
    )
    p.add_argument(
        "--no-rebuild-dist",
        action="store_true",
        help=(
            "Skip the `chelis reef build` step that regenerates "
            "packages/chelis-std/dist/. Useful when chelis itself is "
            "broken mid-bump and you just want the source files updated."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    return p.parse_args(argv)


def validate_version(version: str) -> None:
    if not SEMVER_RE.match(version):
        sys.exit(f"error: {version!r} is not a valid semver string")


def bump_workspace_cargo_toml(version: str, dry_run: bool) -> FileChange | None:
    """Rewrite `[workspace.package] version = "..."` in the root Cargo.toml.

    Uses a line-based rewrite scoped to the `[workspace.package]` section
    so we do not accidentally rewrite a dependency `version = "..."`
    inside `[workspace.dependencies]`.
    """
    text = WORKSPACE_CARGO_TOML.read_text()
    lines = text.splitlines(keepends=True)
    in_section = False
    out: list[str] = []
    found_before: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[workspace.package]"
            out.append(line)
            continue
        if in_section and stripped.startswith("version"):
            # `.*` does not match a trailing newline, so capture and re-emit
            # the newline explicitly to preserve the original line ending.
            m = re.match(r'(\s*version\s*=\s*")([^"]+)(".*?)(\r?\n?)\Z', line)
            if m is not None:
                found_before = m.group(2)
                if found_before != version:
                    out.append(f"{m.group(1)}{version}{m.group(3)}{m.group(4)}")
                    continue
        out.append(line)
    if found_before is None:
        sys.exit(
            f"error: could not find `[workspace.package] version = ...` in {WORKSPACE_CARGO_TOML}"
        )
    if found_before == version:
        return None
    new_text = "".join(out)
    if not dry_run:
        WORKSPACE_CARGO_TOML.write_text(new_text)
    return FileChange(WORKSPACE_CARGO_TOML, found_before, version)


def bump_compiler_pin(path: Path, version: str, dry_run: bool) -> FileChange | None:
    """Rewrite `package.compiler = "=<version>"` in a real .toml file."""
    text = path.read_text()
    expected_after = f"={version}"
    pattern = re.compile(r'^(\s*compiler\s*=\s*")([^"]+)(".*)$', re.MULTILINE)
    m = pattern.search(text)
    if m is None:
        sys.exit(f"error: no `compiler = \"...\"` line in {path}")
    before = m.group(2)
    if before == expected_after:
        return None
    new_text = pattern.sub(
        lambda mm: f"{mm.group(1)}{expected_after}{mm.group(3)}",
        text,
        count=1,
    )
    if not dry_run:
        path.write_text(new_text)
    return FileChange(path, before, expected_after)


def find_chelis_binary() -> Path | None:
    """Look for a `chelis` binary in standard target dirs.

    Returns the first hit in order: `target/release`, `target/debug`.
    Returns None if neither exists; the caller may want to suggest
    `cargo build` first.
    """
    candidates = [
        REPO_ROOT / "target/release/chelis",
        REPO_ROOT / "target/debug/chelis",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def rebuild_chelis_std_dist(dry_run: bool) -> None:
    """Rebuild `packages/chelis-std/dist/*.chb` and `*.tar.zst`.

    Uses `chelis reef build` against the local chelis binary. If no
    chelis binary is present in `target/`, the user must build it first.
    """
    if dry_run:
        print(f"[dry-run] would rebuild {CHELIS_STD_DIST}/")
        return

    chelis = find_chelis_binary()
    if chelis is None:
        sys.exit(
            "error: no chelis binary found in target/release/ or target/debug/. "
            "Run `cargo build --workspace` first, then re-run this script."
        )
    print(f"Rebuilding {CHELIS_STD_DIST.relative_to(REPO_ROOT)} via {chelis} ...")
    cmd = [str(chelis), "reef", "build"]
    result = subprocess.run(cmd, cwd=CHELIS_STD_DIR, check=False)
    if result.returncode != 0:
        sys.exit(
            f"error: `chelis reef build` exited {result.returncode} in "
            f"{CHELIS_STD_DIR}. Fix the build, then re-run this script "
            "with `--no-rebuild-dist` skipped."
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    validate_version(args.version)

    changes: list[FileChange] = []

    cargo_change = bump_workspace_cargo_toml(args.version, args.dry_run)
    if cargo_change is not None:
        changes.append(cargo_change)

    for toml_path in PINNED_REAL_TOML_FILES:
        if not toml_path.exists():
            sys.exit(f"error: pinned-toml file is missing: {toml_path}")
        ch = bump_compiler_pin(toml_path, args.version, args.dry_run)
        if ch is not None:
            changes.append(ch)

    if not changes:
        print(f"All pins already at {args.version}; nothing to do.")
        if not args.no_rebuild_dist and not args.dry_run:
            # Still rebuild dist so the artifacts re-emit even when the
            # source pins were already correct (covers re-running after
            # a partial-state failure).
            rebuild_chelis_std_dist(args.dry_run)
        return 0

    print(f"Bumped to {args.version}:")
    for ch in changes:
        print(ch.render())

    if args.no_rebuild_dist:
        print("(skipped chelis-std dist rebuild — pass without --no-rebuild-dist to regenerate)")
    else:
        rebuild_chelis_std_dist(args.dry_run)

    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
