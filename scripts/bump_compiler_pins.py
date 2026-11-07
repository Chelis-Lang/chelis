#!/usr/bin/env python3
"""Bump the compiler and its explicitly selected Clarabel package identity.

Background: when `workspace.package.version` in the root `Cargo.toml`
changes, its coupled release inputs must change with it:

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
   fixture (`crates/chelis-checker-core/tests/compile_fail/checkpoint_raw_offset/`,
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
   re-resolves the fixture. The root `Cargo.lock` records the workspace
   crates at the workspace version the same way, and CI builds with
   `--locked`, so it is re-resolved here too rather than left as a manual
   `cargo update --workspace` step for the release author.

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

7. The committed crate2nix graph `Cargo.nix`, which the Nix flake imports.
   It records each workspace crate at the workspace version, and the
   `Cargo.nix freshness` CI job fails on any byte difference from an exact
   regeneration. The bump re-runs the crate2nix command that the file's
   header records, through the Nixpkgs that `flake.lock` pins, after the
   root `Cargo.lock` moves. It therefore needs Nix on `PATH`.

8. The carried Clarabel package version. Its native provider admits that exact
   identity, while the standalone package's compiler pin is exact and published
   package assets are immutable. Select the matching standalone release version
   explicitly with --clarabel-version. Before any write, compare it with the
   carried identity in the local previous compiler tag; a new compiler needs a
   newer package identity. An already allocated newer identity is retained, and
   an exact retry regenerates derived artifacts without incrementing it again.
   This script never edits or publishes the standalone repository.

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
  - `test_bump_compiler_pins.py::test_committed_graph_records_every_workspace_crate_at_the_workspace_version`
    names a stale `Cargo.nix` (category 7) before the `Cargo.nix freshness`
    job reports its byte difference.

Usage:
    .venv/bin/python scripts/bump_compiler_pins.py 0.3.2 --clarabel-version 0.1.2

The script is safe to run from any cwd: paths are resolved relative to
the script's own location.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real `.toml` files whose `package.compiler` pin must equal `=<new-version>`.
# Keep in sync with the same list in `compiler_pin_tripwire.rs`.
PINNED_REAL_TOML_FILES: list[Path] = [
    REPO_ROOT / "packages/chelis-clarabel/reef.toml",
    REPO_ROOT / "packages/chelis-std/reef.toml",
    REPO_ROOT / "crates/chelis-cli/tests/fixtures/pseudo_nautilus/reef.toml",
    REPO_ROOT / "crates/chelis-cli/tests/fixtures/release_pipe_stage/reef.toml",
    REPO_ROOT / "examples/illustrative/io_pipeline/reef.toml",
    REPO_ROOT / "examples/clarabel_qp/reef.toml",
    REPO_ROOT / "examples/nautilus_quantile_contract/reef.toml",
    REPO_ROOT / "examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml",
]

WORKSPACE_CARGO_TOML = REPO_ROOT / "Cargo.toml"
CLARABEL_MANIFEST = REPO_ROOT / "packages/chelis-clarabel/reef.toml"

# Category 6: the conformance assets the binary embeds for `conform sync`.
REGEN_CONFORMANCE_ASSETS_SCRIPT = REPO_ROOT / "scripts/regenerate_conformance_assets.py"
CONFORMANCE_ASSETS = REPO_ROOT / "crates/chelis-conformance/assets"

# Category 7: the committed crate2nix graph the Nix flake imports. Its
# header records the exact generation command (see docs/contributor_setup.md).
CARGO_NIX = REPO_ROOT / "Cargo.nix"
CRATE2NIX_HEADER = re.compile(
    r"\A\n# This file was @generated by crate2nix \S+ with the command:\n"
    r"#   (?P<command>[^\n]+)\n"
)

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
    REPO_ROOT / "crates/chelis-checker-core/tests/compile_fail/checkpoint_raw_offset/Cargo.toml",
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
        "--clarabel-version",
        required=True,
        help="Exact Clarabel package version the new compiler must admit; select a fresh release identity.",
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


def release_version_key(version: str) -> tuple:
    """SemVer precedence, excluding build metadata from release ordering."""
    number = r"(?:0|[1-9][0-9]*)"
    match = re.fullmatch(
        rf"({number})\.({number})\.({number})(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?",
        version,
    )
    if match is None:
        sys.exit(f"error: {version!r} is not a valid release version")
    pre, build = match.group(4), match.group(5)
    for value in (pre, build):
        if value is not None and any(not part for part in value.split(".")):
            sys.exit(f"error: {version!r} is not a valid release version")
    identifiers = []
    for part in pre.split(".") if pre else []:
        if part.isdigit():
            if len(part) > 1 and part.startswith("0"):
                sys.exit(f"error: {version!r} has a leading-zero prerelease identifier")
            identifiers.append((0, int(part)))
        else:
            identifiers.append((1, part))
    return (*map(int, match.group(1, 2, 3)), pre is None, tuple(identifiers))


def released_clarabel_version(compiler: str) -> str:
    """Read the immutable local compiler tag; never select versions over the network."""
    reference = f"refs/tags/v{compiler}"
    resolved = subprocess.run(
        ["git", "show-ref", "--verify", "--hash", reference],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    object_id = resolved.stdout.strip()
    if resolved.returncode or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None:
        sys.exit(f"error: cannot read release tag v{compiler}; fetch that tag before bumping")
    revision = f"{object_id}:packages/chelis-clarabel/reef.toml"
    result = subprocess.run(
        ["git", "show", revision], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode:
        sys.exit(f"error: cannot read Clarabel manifest from release tag v{compiler}")
    try:
        package = tomllib.loads(result.stdout)["package"]
        if package["name"] != "chelis-clarabel" or package["compiler"] != f"={compiler}":
            raise ValueError("package name or compiler pin does not match the release tag")
        version = package["version"]
        release_version_key(version)
    except (KeyError, TypeError, ValueError) as error:
        sys.exit(f"error: invalid Clarabel release identity in {revision}: {error}")
    return version


def validate_clarabel_release_pair(compiler: str, target: str) -> None:
    """Reject omitted, reused or regressing identities before the first write."""
    desired_key = release_version_key(compiler)
    target_key = release_version_key(target)
    try:
        workspace = tomllib.loads(WORKSPACE_CARGO_TOML.read_text())["workspace"]["package"]["version"]
        package = tomllib.loads(CLARABEL_MANIFEST.read_text())["package"]
        if package["name"] != "chelis-clarabel":
            raise ValueError("expected the carried chelis-clarabel package")
        current = package["version"]
        pin = package["compiler"]
        if not isinstance(pin, str) or not pin.startswith("="):
            raise ValueError("expected an exact compiler pin")
        previous_compiler = pin[1:]
    except (OSError, KeyError, TypeError, ValueError) as error:
        sys.exit(f"error: invalid compiler/Clarabel release manifests: {error}")
    if desired_key < release_version_key(workspace):
        sys.exit(f"error: compiler target {compiler} is older than {workspace}")
    if target_key < release_version_key(current):
        sys.exit(f"error: Clarabel target {target} is older than allocated package {current}")
    if workspace == compiler and previous_compiler == compiler:
        if target != current:
            sys.exit("error: changing the Clarabel provider identity requires a new compiler version")
        return  # Exact retry, including regeneration after a derived-artifact failure.
    # The workspace may already have moved when an earlier invocation failed.
    baseline_compiler = previous_compiler if workspace == compiler else workspace
    release_version_key(baseline_compiler)
    if desired_key <= release_version_key(baseline_compiler):
        sys.exit("error: the Clarabel compiler pin must advance to a new compiler version")
    baseline = released_clarabel_version(baseline_compiler)
    if target_key <= release_version_key(baseline):
        sys.exit(f"error: Clarabel target {target} must be newer than released package {baseline}")


def bump_clarabel_package_version(version: str, dry_run: bool) -> FileChange | None:
    """Change only the carried manifest's package.version, never dependency versions."""
    text = CLARABEL_MANIFEST.read_text()
    before = tomllib.loads(text)["package"]["version"]
    if before == version:
        return None
    in_package = False
    lines = []
    replacements = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("["):
            in_package = line.split("#", 1)[0].strip() == "[package]"
        if in_package and re.match(r"\s*version\s*=", line):
            line, count = re.subn(r'^(\s*version\s*=\s*")[^"]+(".*)', lambda m: f'{m[1]}{version}{m[2]}', line)
            replacements += count
        lines.append(line)
    if replacements != 1:
        sys.exit("error: expected exactly one carried Clarabel package.version assignment")
    if not dry_run:
        CLARABEL_MANIFEST.write_text("".join(lines))
    return FileChange(CLARABEL_MANIFEST, before, version)


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


def regenerate_workspace_lock(dry_run: bool) -> None:
    """Re-resolve the root `Cargo.lock` after the workspace version moves.

    The same `cargo update --workspace` as the fixture locks: it rewrites only
    the workspace crates' version lines, so a release change set carries no
    registry dependency drift.
    """
    lock = WORKSPACE_CARGO_TOML.with_name("Cargo.lock")
    if dry_run:
        print(f"[dry-run] would regenerate {lock.relative_to(REPO_ROOT)}")
        return
    print(f"Regenerating {lock.relative_to(REPO_ROOT)} via cargo update ...")
    rc = subprocess.run(
        [
            "cargo",
            "update",
            "--workspace",
            "--manifest-path",
            str(WORKSPACE_CARGO_TOML),
        ],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(
            f"error: `cargo update --workspace` exited {rc} for the workspace. "
            f"{lock.relative_to(REPO_ROOT)} is stale, so every `--locked` build "
            "of the release change set will refuse it."
        )


def crate2nix_generation_args(cargo_nix: str) -> list[str]:
    """Return the crate2nix arguments that the `Cargo.nix` header records."""
    header = CRATE2NIX_HEADER.match(cargo_nix)
    if header is None:
        sys.exit(
            f"error: {CARGO_NIX.relative_to(REPO_ROOT)} does not start with the "
            "crate2nix generation header, so the bump cannot regenerate it."
        )
    return shlex.split(header.group("command"))


def regenerate_cargo_nix(dry_run: bool) -> None:
    """Regenerate the committed crate2nix graph after the root lock moves.

    The command comes from the file's own header, which the flake contract
    test and the `Cargo.nix freshness` job hold to the canonical arguments.
    """
    arguments = crate2nix_generation_args(CARGO_NIX.read_text(encoding="utf-8"))
    command = ["nix", "run", "--inputs-from", ".", "nixpkgs#crate2nix", "--", *arguments]
    if dry_run:
        print(
            f"[dry-run] would regenerate {CARGO_NIX.relative_to(REPO_ROOT)} "
            f"via {shlex.join(command)}"
        )
        return
    print(f"Regenerating {CARGO_NIX.relative_to(REPO_ROOT)} via crate2nix ...")
    try:
        rc = subprocess.run(command, cwd=REPO_ROOT).returncode
    except FileNotFoundError:
        rc = 127
    if rc != 0:
        sys.exit(
            f"error: `{shlex.join(command)}` exited {rc}. "
            f"{CARGO_NIX.relative_to(REPO_ROOT)} is stale, so the release pull "
            "request fails the `Cargo.nix freshness` check. Enter Devenv for Nix."
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
    validate_clarabel_release_pair(args.version, args.clarabel_version)

    changes: list[FileChange] = []

    clarabel_change = bump_clarabel_package_version(args.clarabel_version, args.dry_run)
    if clarabel_change is not None:
        changes.append(clarabel_change)

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

    regenerate_workspace_lock(args.dry_run)
    regenerate_cargo_nix(args.dry_run)
    regenerate_compile_fail_fixture_locks(args.dry_run)
    regenerate_conformance_assets(args.dry_run)

    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
