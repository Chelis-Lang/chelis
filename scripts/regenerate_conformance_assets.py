#!/usr/bin/env python3
"""Regenerate agent skill surfaces and embedded conformance assets.

The `chelis-conformance` crate embeds canonical content into the chelis binary
at compile time (`include_str!`), mirroring `chelis-std-bundle`. The repo files
under `agent-skills/` and the root `AGENTS.md` are the source of truth for the
*content*; the committed
copies under `crates/chelis-conformance/assets/` are generated copies.
`.claude/skills` and `.codex/skills` are generated symlinks to the one authored
`agent-skills/` tree. This script rebuilds both forms so the surfaces cannot
drift by hand.

`crates/chelis-conformance/tests/asset_drift_tripwire.rs` asserts the embedded
bytes equal the live repo files, so a forgotten re-run fails the build with a
pointer back here.

Usage:
    python3 scripts/regenerate_conformance_assets.py [--check]

--check exits non-zero (without writing) if the assets are already stale, for a
CI guard that does not want to mutate the tree.

stdlib-only, per the repository scripting policy.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

# The shared skill set the contract (§8) vendors downstream. Keep in lockstep
# with `chelis_conformance::skills::SHARED_SKILLS` and the `agent-skills/`
# directory; the `embedded_skills_match_repo` tripwire locks all three together.
SHARED_SKILLS = [
    "backend-numerics",
    "cli-surface",
    "example-corpus",
    "issue-resolution",
    "packaging-install",
    "phase-gate",
    "redteam-exec",
    "spec-sync",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def planned_skill_copies(root: Path) -> list[tuple[Path, Path]]:
    """Return (src, dest) pairs for every file that should be embedded."""
    src_root = root / "agent-skills"
    dest_root = root / "crates" / "chelis-conformance" / "assets" / "skills"
    pairs: list[tuple[Path, Path]] = []
    for skill in SHARED_SKILLS:
        src_dir = src_root / skill
        if not src_dir.is_dir():
            raise SystemExit(
                f"error: shared skill {skill!r} is not a directory at {src_dir}. "
                f"Update SHARED_SKILLS (and the Rust mirror) if the skill set changed."
            )
        # Lockstep with the Rust consumer: skills.rs embeds, scaffold.rs
        # materializes, and audit.rs drift-checks ONLY `SKILL.md`. If this
        # generator embedded extra files, they would be dead weight the shell
        # never receives and the audit never checks. Refuse a multi-file skill
        # until the Rust side is taught to handle it.
        files = sorted(p for p in src_dir.rglob("*") if p.is_file())
        skill_md = src_dir / "SKILL.md"
        if files != [skill_md]:
            found = [str(p.relative_to(src_dir)) for p in files]
            raise SystemExit(
                f"error: shared skill {skill!r} must contain exactly one file, SKILL.md "
                f"(found {found}). The Rust embed/materialize/audit path only handles "
                f"SKILL.md; teach skills.rs, scaffold.rs, and audit.rs about the extra "
                f"files before this generator may embed them."
            )
        pairs.append((skill_md, dest_root / skill_md.relative_to(src_root)))
    return pairs


def agent_surface_layout_reasons(root: Path) -> list[str]:
    """Require both agent discovery paths to link to the authored skill tree."""
    reasons = []
    for surface in (root / ".claude" / "skills", root / ".codex" / "skills"):
        if not surface.is_symlink():
            reasons.append(f"agent skill surface is not a symlink: {surface}")
            continue
        try:
            target = surface.readlink()
        except OSError as exc:
            reasons.append(f"cannot read agent skill surface symlink {surface}: {exc}")
            continue
        if target != Path("../agent-skills"):
            reasons.append(
                f"agent skill surface {surface} points to {target}, expected ../agent-skills"
            )
    return reasons


def materialize_agent_surface_links(root: Path) -> None:
    """Replace legacy copies or pointers with the canonical discovery symlinks."""
    for surface in (root / ".claude" / "skills", root / ".codex" / "skills"):
        if surface.is_symlink() or surface.is_file():
            surface.unlink()
        elif surface.exists():
            shutil.rmtree(surface)
        surface.parent.mkdir(parents=True, exist_ok=True)
        surface.symlink_to("../agent-skills", target_is_directory=True)


def planned_canonical_copies(root: Path) -> list[tuple[Path, Path]]:
    """Return canonical documents whose embedded bytes mirror repo files."""
    return [
        (
            root / "AGENTS.md",
            root
            / "crates"
            / "chelis-conformance"
            / "assets"
            / "canonical"
            / "agents-inheritance.md",
        )
    ]


def is_stale(pairs: list[tuple[Path, Path]], dest_root: Path) -> list[str]:
    """Return a list of human-readable reasons the embedded copy is stale."""
    reasons: list[str] = []
    expected_dests = {dest for _, dest in pairs}
    # Missing or differing files.
    for src, dest in pairs:
        if not dest.exists():
            reasons.append(f"missing embedded copy: {dest}")
        elif not filecmp.cmp(src, dest, shallow=False):
            reasons.append(f"content differs: {dest}")
    # Stale files that no longer have a source (removed upstream skill/file).
    if dest_root.exists():
        for existing in sorted(dest_root.rglob("*")):
            if existing.is_file() and existing not in expected_dests:
                reasons.append(f"orphaned embedded copy (no source): {existing}")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if assets are stale; do not write",
    )
    args = parser.parse_args()

    root = repo_root()
    dest_skills = root / "crates" / "chelis-conformance" / "assets" / "skills"
    skill_pairs = planned_skill_copies(root)
    canonical_pairs = planned_canonical_copies(root)
    pairs = skill_pairs + canonical_pairs

    if args.check:
        reasons = is_stale(pairs, dest_skills) + agent_surface_layout_reasons(root)
        if reasons:
            print("conformance assets are STALE:", file=sys.stderr)
            for r in reasons:
                print(f"  - {r}", file=sys.stderr)
            print(
                "Run `python3 scripts/regenerate_conformance_assets.py` and commit "
                "crates/chelis-conformance/assets/.",
                file=sys.stderr,
            )
            return 1
        print("conformance assets are up to date.")
        return 0

    # Rebuild from scratch so a removed upstream file cannot linger.
    reasons_before = is_stale(pairs, dest_skills) + agent_surface_layout_reasons(root)
    if dest_skills.exists():
        shutil.rmtree(dest_skills)
    for src, dest in skill_pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    for src, dest in canonical_pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    materialize_agent_surface_links(root)

    if reasons_before:
        print(f"regenerated {len(pairs)} embedded conformance file(s); changes:")
        for r in reasons_before:
            print(f"  - {r}")
    else:
        print(f"embedded {len(pairs)} conformance file(s); already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
