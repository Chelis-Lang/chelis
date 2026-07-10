#!/usr/bin/env python3
"""Bump the workspace version and every coupled compiler pin in lockstep.

Background: when `workspace.package.version` in the root `Cargo.toml`
changes, five categories of files must change with it:

1. Test fixtures with hardcoded `compiler = "=X.Y.Z"` strings. These are
   already auto-synced via `chelis_compiler_api::COMPILER_VERSION`
   (which uses `env!("CARGO_PKG_VERSION")` at compile time). This script
   does NOT touch them. The auto-sync handles the work.

2. Real `.toml` files that ship in the repo and must hand-pin a compiler
   version. The integration tests installs `chelis-std` from
   `packages/chelis-std/` and compares its embedded `package.compiler`
   pin against the running binary's version, so a stale pin fails CI.

3. The prebuilt `packages/chelis-std/dist/*.chb` and `*.tar.zst` artifacts.
   These are produced by `chelis reef build` and embed the bumped pin.

4. The compile-time-embedded bundle in `crates/chelis-std-bundle/dist/`.
   These bytes are baked into the chelis binary via `include_bytes!()` and
   carry chelis-std's `reef.toml` (with its `compiler =` pin) inside the
   archive. The reef loader reads that embedded `reef.toml` through
   `validate_manifest`, which rejects any pin other than the running
   compiler's, so a stale embedded bundle makes every chelis-std-importing
   program fail at load. They are a copy of category (3), kept in their own
   crate dir. This is the 0.9.0 release failure: categories (2) and (3)
   were bumped but the embedded bundle was never regenerated, so the
   release-only fixture step rejected it.

5. Committed `reef.lock` files that record a `chelis-std` bundled
   dependency (`packages/chelis-std/reef.lock` and the
   `release_pipe_stage` fixture). Their dependency `compiler =` pin and
   `archive_sha256`/`shell_sha256` are synthesized from the embedded
   bundle, so they go stale the moment category (4) is regenerated. They
   are NOT auto-synced — they are literal on-disk strings.

6. The Hull conformance corpus manifest
   (`tests/conformance/hull/manifest.json`, `chelis_version_pinned`). The
   conformance gate's STALE CORPUS check compares this pin to the live
   binary's version, so leaving it behind turns the Hull Conformance
   workflow red on main the moment the release merges (the 0.15.0
   failure: the pin was a separate post-release chore, opening a red
   window per release). Bumping it here is sound because the release
   PR's own conformance-gate run then validates the frozen corpus
   against the new binary — a real behavior change in the compiler
   still fails that PR loudly.

This script is the single, scriptable entry point for the release bump.
Two tripwire tests fail loudly when these drift, pointing future operators
at this script:
  - `compiler_pin_tripwire.rs::real_toml_compiler_pins_match_workspace_version`
    (category 2 vs 1)
  - `compiler_pin_tripwire.rs::real_lock_compiler_pins_match_workspace_version`
    (category 5)
  - `chelis-std-bundle::extract_yields_reef_package_layout` asserts the
    embedded bundle pin (category 4).

Usage:
    python3 scripts/bump_compiler_pins.py 0.3.2

The script is safe to run from any cwd: paths are resolved relative to
the script's own location.
"""
from __future__ import annotations

import argparse
import re
import shutil
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

# The compile-time-embedded bundle copied into the binary. Refreshed in
# lockstep with CHELIS_STD_DIST via the canonical bundle pipeline.
BUNDLE_DIST = REPO_ROOT / "crates/chelis-std-bundle/dist"
REGEN_BUNDLE_SCRIPT = REPO_ROOT / "scripts/regenerate_chelis_std_bundle.py"

# The Hull conformance corpus manifest (category 6): its
# `chelis_version_pinned` feeds the gate's STALE CORPUS check against the
# live binary. `test_corpus_integrity.py` asserts it matches the workspace
# version.
HULL_MANIFEST = REPO_ROOT / "tests/conformance/hull/manifest.json"

# Package roots whose committed `reef.lock` records a `chelis-std`
# bundled dependency. Their synthesized pin and sha256s go stale the
# moment the embedded bundle is regenerated, so re-run `chelis reef build`
# in each to rewrite the lock. Keep in sync with `pinned_real_lock_files`
# in `crates/chelis-cli/tests/compiler_pin_tripwire.rs`.
PINNED_REAL_LOCK_DIRS: list[Path] = [
    CHELIS_STD_DIR,
    REPO_ROOT / "crates/chelis-cli/tests/fixtures/release_pipe_stage",
]


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


def bump_hull_manifest_pin(version: str, dry_run: bool) -> FileChange | None:
    """Rewrite `chelis_version_pinned` in the Hull conformance manifest.

    Line-based rewrite (not json round-trip) so the frozen artifact's
    formatting and key order stay byte-stable apart from the pin itself.
    """
    text = HULL_MANIFEST.read_text()
    pattern = re.compile(r'^(\s*"chelis_version_pinned"\s*:\s*")([^"]+)(".*)$', re.MULTILINE)
    m = pattern.search(text)
    if m is None:
        sys.exit(f"error: no `chelis_version_pinned` entry in {HULL_MANIFEST}")
    before = m.group(2)
    if before == version:
        return None
    new_text = pattern.sub(
        lambda mm: f"{mm.group(1)}{version}{mm.group(3)}",
        text,
        count=1,
    )
    if not dry_run:
        HULL_MANIFEST.write_text(new_text)
    return FileChange(HULL_MANIFEST, before, version)


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
    """Rebuild the chelis-std artifacts the binary ships, in lockstep.

    The chelis-std runtime exists as two byte-identical copies:
    `packages/chelis-std/dist/` (the published package) and
    `crates/chelis-std-bundle/dist/` (the bytes baked into the binary via
    `include_bytes!()`). A version bump must refresh BOTH or the embedded
    bundle goes stale and rejects every chelis-std-importing program at
    load — the 0.9.0 release failure.

    Pipeline (delegated to `regenerate_chelis_std_bundle.py`, the canonical
    bundle pipeline):
      1. `cargo build -p chelis-cli --release`
      2. `chelis reef build packages/chelis-std/`  (-> category 3)
      3. copy the result into `crates/chelis-std-bundle/dist/` (category 4)
    Then this function rebuilds the CLI a second time so the binary embeds
    the freshly-copied bundle bytes, and regenerates the committed
    `reef.lock` files (category 5) with that binary so their synthesized
    `chelis-std` pin and `archive_sha256`/`shell_sha256` match the new
    bundle.
    """
    if dry_run:
        print(f"[dry-run] would regenerate {CHELIS_STD_DIST.relative_to(REPO_ROOT)}/")
        print(f"[dry-run] would regenerate {BUNDLE_DIST.relative_to(REPO_ROOT)}/")
        for lock in PINNED_REAL_LOCK_DIRS:
            print(f"[dry-run] would regenerate {(lock / 'reef.lock').relative_to(REPO_ROOT)}")
        return

    # Steps 1-3: build CLI, build chelis-std dist, copy into the bundle
    # crate. Delegated to the canonical bundle pipeline so the two scripts
    # cannot drift on how the embedded bytes are produced.
    print(f"Regenerating chelis-std dist + embedded bundle via {REGEN_BUNDLE_SCRIPT.name} ...")
    rc = subprocess.run(
        [sys.executable, str(REGEN_BUNDLE_SCRIPT)],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(
            f"error: {REGEN_BUNDLE_SCRIPT.name} exited {rc}. Fix the build, "
            "then re-run this script (without --no-rebuild-dist)."
        )

    # The regen script built the CLI BEFORE copying the new bundle bytes,
    # so its binary still embeds the prior bundle. Rebuild once more so the
    # binary's `include_bytes!()` picks up the freshly-copied bytes; the
    # lockfile sha256s come from those embedded bytes, so the binary must be
    # current before it writes a lock.
    print("Rebuilding chelis-cli so the binary embeds the new bundle ...")
    rc = subprocess.run(
        ["cargo", "build", "-p", "chelis-cli", "--release"],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit("error: cargo build -p chelis-cli --release failed after bundle copy")

    chelis = find_chelis_binary()
    if chelis is None:
        sys.exit(
            "error: no chelis binary found in target/release/ or target/debug/ "
            "after rebuild."
        )

    # Step 5: regenerate the committed reef.lock files. `chelis reef build`
    # writes `reef.lock` at the package root, synthesizing the chelis-std
    # bundled dependency from the now-current embedded bundle.
    for pkg_dir in PINNED_REAL_LOCK_DIRS:
        lock = pkg_dir / "reef.lock"
        print(f"Regenerating {lock.relative_to(REPO_ROOT)} via {chelis} reef build ...")
        rc = subprocess.run(
            [str(chelis), "reef", "build"],
            cwd=pkg_dir,
        ).returncode
        if rc != 0:
            sys.exit(
                f"error: `chelis reef build` exited {rc} in {pkg_dir}. "
                "The committed reef.lock could not be regenerated."
            )
        # `reef build` also drops a `dist/` next to the package; for the
        # fixture that dir is not tracked, so leave the source tree clean.
        _clean_untracked_dist(pkg_dir)


def _clean_untracked_dist(pkg_dir: Path) -> None:
    """Remove a `dist/` that `chelis reef build` writes next to a package
    whose committed tree does not track build artifacts.

    Only `packages/chelis-std/dist/` is a tracked artifact dir; every other
    package root (the release fixture) keeps no `dist/`, so the build's
    output is incidental and must not be left behind.
    """
    if pkg_dir == CHELIS_STD_DIR:
        return
    dist = pkg_dir / "dist"
    if dist.is_dir():
        shutil.rmtree(dist)


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

    manifest_change = bump_hull_manifest_pin(args.version, args.dry_run)
    if manifest_change is not None:
        changes.append(manifest_change)

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
        print("(skipped chelis-std dist rebuild. Pass without --no-rebuild-dist to regenerate)")
    else:
        rebuild_chelis_std_dist(args.dry_run)

    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
