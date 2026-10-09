"""Unit tests for `regenerate_conformance_assets.py`.

Run via: `python3 -m unittest scripts.test_regenerate_conformance_assets` from
repo root, or through the CI `unittest discover -s scripts` step.

Locked here:
  (a) REPO_SKILLS matches the live `agent-skills/` directory, and
      SHARED_SKILLS is exactly the repo skills plus the package skills (the
      Python mirror of the Rust `skills::SHARED_SKILLS` tripwire);
  (b) planned_skill_copies enumerates one dest per real source file, under the
      crate's assets/skills/ tree;
  (c) is_stale detects missing, differing, and orphaned embedded copies;
  (d) a real regenerate makes --check pass (round-trip idempotence);
  (e) every managed-block body, the capability surface guide included, is a
      generated copy of an authored repo document (chelis#2831).
"""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "regenerate_conformance_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regen_conformance_assets", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regen = _load_module()
ROOT = regen.repo_root()


class SharedSkillsListTests(unittest.TestCase):
    def test_repo_skills_matches_agent_skills_dir(self):
        on_disk = {
            p.name for p in (ROOT / "agent-skills").iterdir() if p.is_dir()
        }
        self.assertEqual(
            set(regen.REPO_SKILLS) | set(regen.LOCAL_SKILLS),
            on_disk,
            "REPO_SKILLS plus LOCAL_SKILLS in regenerate_conformance_assets.py "
            "disagrees with agent-skills/; update them (and the Rust "
            "skills::SHARED_SKILLS or skills::LOCAL_SKILLS mirror).",
        )

    def test_local_skills_are_never_shared_or_embedded(self):
        self.assertFalse(set(regen.LOCAL_SKILLS) & set(regen.SHARED_SKILLS))
        embedded = {dest.parent.name for _, dest in regen.planned_skill_copies(ROOT)}
        self.assertFalse(set(regen.LOCAL_SKILLS) & embedded)

    def test_shared_skills_are_repo_plus_package_skills(self):
        package = [name for name, _ in regen.PACKAGE_SKILLS]
        self.assertEqual(regen.SHARED_SKILLS, sorted(regen.REPO_SKILLS + package))
        self.assertIn("chelis-std", package)
        self.assertFalse(set(package) & set(regen.REPO_SKILLS))


class PlannedCopiesTests(unittest.TestCase):
    def test_one_dest_per_source_file(self):
        pairs = regen.planned_skill_copies(ROOT)
        # Exactly one SKILL.md per shared skill, the package skill included.
        self.assertEqual(
            sorted(dest.parent.name for _, dest in pairs), regen.SHARED_SKILLS
        )
        self.assertIn(
            (
                ROOT / "packages" / "chelis-std" / "SKILL.md",
                ROOT
                / "crates"
                / "chelis-conformance"
                / "assets"
                / "skills"
                / "chelis-std"
                / "SKILL.md",
            ),
            pairs,
        )
        dest_root = ROOT / "crates" / "chelis-conformance" / "assets" / "skills"
        for src, dest in pairs:
            self.assertTrue(src.is_file(), f"source missing: {src}")
            self.assertTrue(
                str(dest).startswith(str(dest_root)),
                f"dest escapes assets/skills/: {dest}",
            )

    def test_managed_block_bodies_are_embedded_canonical_copies(self):
        pairs = regen.planned_canonical_copies(ROOT)
        dest_root = regen.canonical_dest_root(ROOT)
        self.assertEqual(
            pairs,
            [
                (ROOT / "AGENTS.md", dest_root / "agents-inheritance.md"),
                (
                    ROOT / "docs" / "CHELIS_SURFACE.md",
                    dest_root / "chelis-surface.md",
                ),
            ],
        )

    def test_canonical_dir_holds_only_generated_bodies(self):
        # Every committed canonical body has an authored source. A hand-written
        # body (the retired `chelis-surface-header.md` was one) is an orphan.
        dest_root = regen.canonical_dest_root(ROOT)
        planned = {dest for _, dest in regen.planned_canonical_copies(ROOT)}
        on_disk = {p for p in dest_root.iterdir() if p.is_file()}
        self.assertEqual(on_disk, planned)

    def test_agent_surfaces_must_link_to_the_authored_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".claude").mkdir()
            (root / ".codex").mkdir()
            (root / ".claude/skills").symlink_to("../agent-skills", target_is_directory=True)
            (root / ".codex/skills").symlink_to("../agent-skills", target_is_directory=True)
            self.assertEqual(regen.agent_surface_layout_reasons(root), [])

            (root / ".codex/skills").unlink()
            (root / ".codex/skills").symlink_to("../other", target_is_directory=True)
            reasons = regen.agent_surface_layout_reasons(root)
            self.assertEqual(len(reasons), 1)
            self.assertIn("../agent-skills", reasons[0])

    def test_agent_surface_link_regeneration_replaces_legacy_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".claude/skills/legacy").mkdir(parents=True)
            (root / ".codex").mkdir()
            (root / ".codex/skills").symlink_to("../other", target_is_directory=True)

            regen.materialize_agent_surface_links(root)

            for surface in (root / ".claude/skills", root / ".codex/skills"):
                self.assertTrue(surface.is_symlink(), surface)
                self.assertEqual(surface.readlink(), Path("../agent-skills"))


class IsStaleTests(unittest.TestCase):
    def _fake_tree(self, tmp: Path):
        """Build a minimal agent-skills-shaped source tree + return (pairs, dest)."""
        src_root = tmp / "agent-skills"
        dest_root = tmp / "assets" / "skills"
        skill = src_root / "spec-sync"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("body\n")
        src = skill / "SKILL.md"
        pairs = [(src, dest_root / "spec-sync" / "SKILL.md")]
        return pairs, dest_root

    def test_hand_written_canonical_body_is_an_orphan(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "AGENTS.md"
            src.write_text("contract\n")
            dest_root = tmp / "assets" / "canonical"
            dest_root.mkdir(parents=True)
            pairs = [(src, dest_root / "agents-inheritance.md")]
            shutil.copyfile(src, pairs[0][1])
            self.assertEqual(regen.is_stale(pairs, dest_root), [])

            (dest_root / "chelis-surface-header.md").write_text("authored here\n")
            reasons = regen.is_stale(pairs, dest_root)
            self.assertEqual(len(reasons), 1, reasons)
            self.assertIn("orphaned", reasons[0])

    def test_missing_then_clean_then_diff_then_orphan(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pairs, dest_root = self._fake_tree(tmp)

            # (1) nothing copied yet -> missing.
            reasons = regen.is_stale(pairs, dest_root)
            self.assertTrue(any("missing" in r for r in reasons), reasons)

            # (2) copy exactly -> clean.
            for src, dest in pairs:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            self.assertEqual(regen.is_stale(pairs, dest_root), [])

            # (3) mutate the embedded copy -> differs.
            pairs[0][1].write_text("tampered\n")
            reasons = regen.is_stale(pairs, dest_root)
            self.assertTrue(any("differs" in r for r in reasons), reasons)

            # (4) restore + add an orphan file with no source -> orphaned.
            shutil.copyfile(pairs[0][0], pairs[0][1])
            (dest_root / "spec-sync" / "STALE.md").write_text("x\n")
            reasons = regen.is_stale(pairs, dest_root)
            self.assertTrue(any("orphaned" in r for r in reasons), reasons)


class RoundTripTests(unittest.TestCase):
    def test_repo_assets_are_current(self):
        # The committed assets must already be up to date: the regenerate script
        # in --check mode is a CI-safe guard, and this asserts the checked-in
        # tree matches the live agent-skills/.
        dest_root = ROOT / "crates" / "chelis-conformance" / "assets" / "skills"
        self.assertEqual(
            regen.is_stale(regen.planned_skill_copies(ROOT), dest_root)
            + regen.is_stale(
                regen.planned_canonical_copies(ROOT), regen.canonical_dest_root(ROOT)
            ),
            [],
            "committed conformance assets are stale; run "
            "`python3 scripts/regenerate_conformance_assets.py`.",
        )


class MultiFileSkillRejectedTests(unittest.TestCase):
    """The generator must stay in lockstep with the SKILL.md-only Rust consumer:
    a skill dir with anything besides SKILL.md is rejected, not silently
    embedded (dead weight the shell never receives)."""

    def _one_skill_root(self, tmp: Path, *files: str) -> Path:
        skill = tmp / "agent-skills" / "spec-sync"
        skill.mkdir(parents=True)
        for name in files:
            (skill / name).write_text("body\n")
        return tmp

    def test_extra_file_in_skill_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._one_skill_root(Path(td), "SKILL.md", "EXTRA.md")
            saved = (regen.REPO_SKILLS, regen.PACKAGE_SKILLS)
            regen.REPO_SKILLS, regen.PACKAGE_SKILLS = ["spec-sync"], []
            try:
                with self.assertRaises(SystemExit):
                    regen.planned_skill_copies(root)
            finally:
                regen.REPO_SKILLS, regen.PACKAGE_SKILLS = saved

    def test_single_file_skill_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._one_skill_root(Path(td), "SKILL.md")
            saved = (regen.REPO_SKILLS, regen.PACKAGE_SKILLS)
            regen.REPO_SKILLS, regen.PACKAGE_SKILLS = ["spec-sync"], []
            try:
                pairs = regen.planned_skill_copies(root)
                self.assertEqual(len(pairs), 1)
                self.assertTrue(str(pairs[0][0]).endswith("spec-sync/SKILL.md"))
            finally:
                regen.REPO_SKILLS, regen.PACKAGE_SKILLS = saved


if __name__ == "__main__":
    unittest.main()
