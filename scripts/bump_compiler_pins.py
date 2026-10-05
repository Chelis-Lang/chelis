#!/usr/bin/env python3
"""Bump the workspace version and every coupled compiler pin in lockstep.

Background: when `workspace.package.version` in the root `Cargo.toml`
changes, six categories of files must change with it:

1. Test fixtures with hardcoded `compiler = "=X.Y.Z"` strings. These are
   already auto-synced via `chelis_compiler_api::COMPILER_VERSION`
   (which uses `env!("CARGO_PKG_VERSION")` at compile time). This script
   does NOT touch them. The auto-sync handles the work.

2. Real `.toml` files that ship in the repo and must hand-pin a compiler
   version. `validate_manifest` rejects any pin other than the running
   binary's version, so a stale pin fails CI.
   `packages/chelis-std/reef.toml` is among them: the chelis-std runtime each
   binary embeds is packed from that tree by
   `crates/chelis-std-bundle/build.rs` while the compiler builds, so the
   bumped pin reaches the embedded runtime with no committed artifact or
   lock to refresh.

3. The Hull conformance corpus manifest
   (`tests/conformance/hull/manifest.json`, `chelis_version_pinned`). The
   conformance gate's STALE CORPUS check compares this pin to the live
   binary's version, so leaving it behind turns the Hull Conformance
   workflow red on main the moment the release merges (the 0.15.0
   failure: the pin was a separate post-release chore, opening a red
   window per release). Bumping it here is sound because the release
   PR's own conformance-gate run then validates the frozen corpus
   against the new binary — a real behavior change in the compiler
   still fails that PR loudly.

4. The committed `Cargo.lock` beside each out-of-workspace compile-fail
   fixture (`crates/chelis-types/tests/compile_fail/checkpoint_raw_offset/`
   `crates/chelis-compiler-api/tests/compile_fail/pipeline_artifacts/`, and
   `crates/chelis-unord/tests/compile_fail/order_escape/`).
   Each fixture is its own one-crate workspace that depends on the real
   crates by path, so its lock records them at the workspace version. They
   are compiled by a `gate.py` step with `cargo check --locked`, which
   refuses to update a stale lock:

       error: cannot update the lock file ... because --locked was passed

   The gate step then reports the fixture's *diagnostics* as missing, which
   reads as a compile-fail regression rather than a stale lock — a slow
   thing to diagnose under release pressure, which is exactly when it fires
   (the 0.18.2 failure, chelis#1128; the pipeline-artifacts sibling is
   chelis#1234, and its message names six phantom regressions at once). The
   locks are NOT auto-synced: cargo writes them, but only when something
   re-resolves the fixture.

5. Checked-in test-fixture data that hand-pins the compiler. A Rust
   fixture built from a string literal auto-syncs through category (1),
   but a fixture that lives on disk as `.toml`/`.json` data cannot: it is
   `include_str!`'d verbatim and written to a temp package, where
   `validate_manifest` rejects any pin but the running binary's. The
   `crates/chelis-reef/tests/fixtures/pipeline_parity/` set is the
   instance -- two `reef.toml` manifests plus the frozen
   `expected_schema.json` / `expected_shell.json` captures that record the
   same pin. It arrived after the 0.18.4 bump, went stale at 0.18.5 (the
   active rejected leg failed with `package.compiler must be =0.18.5`
   where it expected the frozen type errors), and was hand-repaired by the
   release operator both times. `expected_hashes.txt` is deliberately NOT
   in this set: `BASELINE.md` records those values as nondeterministic
   across machines and no longer asserted, and the leg that reads them is
   `#[ignore]`d pending chelis#1198.

6. The conformance assets under `crates/chelis-conformance/assets/`: the
   embedded copies of the root `AGENTS.md`, `docs/CHELIS_SURFACE.md`, and the
   shared skills (`agent-skills/*/SKILL.md` and `packages/chelis-std/SKILL.md`)
   that `chelis reef conform sync` materializes into shells. They carry no
   version (the link pinning happens in `sync`, against the shell's pin), but
   a release ships whatever bytes are embedded, so a release PR whose sources
   moved since the last regeneration would ship a stale agent surface to every
   shell. `regenerate_conformance_assets.py` rebuilds them from their sources;
   `asset_drift_tripwire.rs` is the guard.

This script is the single, scriptable entry point for the release bump.
These tripwires fail loudly when the categories drift, pointing future
operators at this script:
  - `compiler_pin_tripwire.rs::real_toml_compiler_pins_match_workspace_version`
    (category 2 vs 1)
  - `chelis-std-bundle::extract_yields_reef_package_layout` asserts the
    embedded runtime's pin equals the workspace version.
  - `scripts/check_checkpoint_compile_fail.py` and
    `scripts/check_pipeline_core_compile_fail.py`, both `gate.py` stages,
    fail on a stale fixture lock (category 4) — but blame the fixture's
    diagnostics, not the lock, which is why
    `test_bump_compiler_pins.py::test_each_lock_pins_every_path_package_to_the_workspace_version`
    exists to name the real cause first.

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
    REPO_ROOT / "examples/illustrative/io_pipeline/reef.toml",
    REPO_ROOT / "examples/nautilus_quantile_contract/reef.toml",
    REPO_ROOT / "examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml",
]

WORKSPACE_CARGO_TOML = REPO_ROOT / "Cargo.toml"

# Category 6: the conformance assets the binary embeds for `conform sync`.
REGEN_CONFORMANCE_ASSETS_SCRIPT = REPO_ROOT / "scripts/regenerate_conformance_assets.py"
CONFORMANCE_ASSETS = REPO_ROOT / "crates/chelis-conformance/assets"

# The Hull conformance corpus manifest (category 3): its
# `chelis_version_pinned` feeds the gate's STALE CORPUS check against the
# live binary. `test_corpus_integrity.py` asserts it matches the workspace
# version.
HULL_MANIFEST = REPO_ROOT / "tests/conformance/hull/manifest.json"

# Checked-in fixture data that hand-pins the compiler (category 5). These
# are `include_str!`'d by their tests and written verbatim to a temp
# package, so they cannot auto-sync the way a Rust string literal does.
# `expected_hashes.txt` is deliberately absent: those values are
# nondeterministic across machines and the leg reading them is `#[ignore]`d
# (chelis#1198).
PIPELINE_PARITY_FIXTURES = (
    REPO_ROOT / "crates/chelis-reef/tests/fixtures/pipeline_parity"
)
PINNED_FIXTURE_TOML_FILES: list[Path] = [
    PIPELINE_PARITY_FIXTURES / "accepted/reef.toml",
    PIPELINE_PARITY_FIXTURES / "rejected/reef.toml",
]
PINNED_FIXTURE_JSON_FILES: list[Path] = [
    PIPELINE_PARITY_FIXTURES / "accepted/expected_schema.json",
    PIPELINE_PARITY_FIXTURES / "accepted/expected_shell.json",
]

# Manifests of the out-of-workspace compile-fail fixtures (category 4).
# Each is its own one-crate workspace depending on the real crates by path,
# so its committed sibling `Cargo.lock` records them at the workspace
# version. Their gate steps compile them with `cargo check --locked`, which
# refuses to update a stale lock. Keep in sync with the `MANIFEST` constant
# in the matching `scripts/check_*_compile_fail.py`.
COMPILE_FAIL_FIXTURE_MANIFESTS: list[Path] = [
    REPO_ROOT / "crates/chelis-types/tests/compile_fail/checkpoint_raw_offset/Cargo.toml",
    REPO_ROOT / "crates/chelis-compiler-api/tests/compile_fail/pipeline_artifacts/Cargo.toml",
    REPO_ROOT / "crates/chelis-unord/tests/compile_fail/order_escape/Cargo.toml",
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
            "pin, then regenerate the derived artifacts that record them."
        ),
    )
    p.add_argument(
        "version",
        help="New workspace version, e.g. 0.3.2 (semver)",
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


def bump_json_compiler_pin(path: Path, version: str, dry_run: bool) -> FileChange | None:
    """Rewrite a `"compiler": "=<version>"` field in a frozen JSON capture.

    Line-based rewrite (not a json round-trip) for the same reason
    `bump_hull_manifest_pin` uses one: these captures are compared
    byte-for-byte against `serde_json::to_string_pretty` output, so only
    the pin may move.
    """
    text = path.read_text()
    expected_after = f"={version}"
    pattern = re.compile(r'^(\s*"compiler"\s*:\s*")([^"]+)(".*)$', re.MULTILINE)
    m = pattern.search(text)
    if m is None:
        sys.exit(f'error: no `"compiler": "..."` field in {path}')
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


def regenerate_compile_fail_fixture_locks(dry_run: bool) -> None:
    """Re-resolve the committed lock beside each compile-fail fixture.

    `cargo update --workspace` re-resolves only the local path packages —
    it rewrites the four-or-so `chelis-*` version lines the workspace bump
    just invalidated and leaves every registry pin and checksum alone. That
    matters: `cargo generate-lockfile` would drag unrelated dependencies
    forward inside a release change set.

    It re-resolves rather than compiles, so it needs no working `chelis`
    binary.
    """
    for manifest in COMPILE_FAIL_FIXTURE_MANIFESTS:
        lock = manifest.with_name("Cargo.lock")
        if dry_run:
            print(f"[dry-run] would regenerate {lock.relative_to(REPO_ROOT)}")
            continue
        if not manifest.is_file():
            sys.exit(f"error: compile-fail fixture manifest is missing: {manifest}")
        print(f"Regenerating {lock.relative_to(REPO_ROOT)} via cargo update ...")
        rc = subprocess.run(
            [
                "cargo",
                "update",
                "--workspace",
                "--manifest-path",
                str(manifest),
            ],
            cwd=REPO_ROOT,
        ).returncode
        if rc != 0:
            sys.exit(
                f"error: `cargo update --workspace` exited {rc} for {manifest}. "
                f"{lock.relative_to(REPO_ROOT)} is stale, so its compile-fail "
                "gate step will fail with a missing-diagnostic message that "
                "does not name the real cause."
            )


def regenerate_conformance_assets(dry_run: bool) -> None:
    """Rebuild the embedded conformance assets from their sources (category 6).

    It copies files and needs no `chelis` binary. It is version-independent;
    shells get pinned links at sync time.
    """
    if dry_run:
        print(
            f"[dry-run] would regenerate {CONFORMANCE_ASSETS.relative_to(REPO_ROOT)}/ "
            f"via {REGEN_CONFORMANCE_ASSETS_SCRIPT.relative_to(REPO_ROOT)}"
        )
        return
    print(f"Regenerating {CONFORMANCE_ASSETS.relative_to(REPO_ROOT)}/ ...")
    rc = subprocess.run(
        [sys.executable, str(REGEN_CONFORMANCE_ASSETS_SCRIPT)],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(
            f"error: {REGEN_CONFORMANCE_ASSETS_SCRIPT.relative_to(REPO_ROOT)} exited {rc}. "
            "The release would embed a stale AGENTS.md, CHELIS_SURFACE.md, or shared "
            "skill set for `chelis reef conform sync`."
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

    for toml_path in PINNED_FIXTURE_TOML_FILES:
        if not toml_path.exists():
            sys.exit(f"error: pinned fixture toml is missing: {toml_path}")
        ch = bump_compiler_pin(toml_path, args.version, args.dry_run)
        if ch is not None:
            changes.append(ch)

    for json_path in PINNED_FIXTURE_JSON_FILES:
        if not json_path.exists():
            sys.exit(f"error: pinned fixture json is missing: {json_path}")
        ch = bump_json_compiler_pin(json_path, args.version, args.dry_run)
        if ch is not None:
            changes.append(ch)

    manifest_change = bump_hull_manifest_pin(args.version, args.dry_run)
    if manifest_change is not None:
        changes.append(manifest_change)

    if changes:
        print(f"Bumped to {args.version}:")
        for ch in changes:
            print(ch.render())
    else:
        # Still regenerate the derived artifacts even when the source pins
        # were already correct: that is the state a re-run after a
        # partial-state failure lands in, and it is exactly when a derived
        # artifact is the thing left stale.
        print(f"All pins already at {args.version}; nothing to do.")

    regenerate_compile_fail_fixture_locks(args.dry_run)
    regenerate_conformance_assets(args.dry_run)

    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
