"""Unit tests for `regenerate_conformance_assets.py`.

Run via: `python3 -m unittest scripts.test_regenerate_conformance_assets` from
repo root, or through the CI `unittest discover -s scripts` step.

Locked here:
  (a) SHARED_SKILLS matches the live `agent-skills/` directory (the Python
      mirror of the Rust `skills::SHARED_SKILLS` tripwire);
  (b) planned_skill_copies enumerates one dest per real source file, under the
      crate's assets/skills/ tree;
  (c) is_stale detects missing, differing, and orphaned embedded copies;
  (d) a real regenerate makes --check pass (round-trip idempotence).
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
    def test_shared_skills_matches_agent_skills_dir(self):
        on_disk = {
            p.name for p in (ROOT / "agent-skills").iterdir() if p.is_dir()
        }
        self.assertEqual(
            set(regen.SHARED_SKILLS),
            on_disk,
            "SHARED_SKILLS in regenerate_conformance_assets.py disagrees with "
            "agent-skills/; update it (and the Rust skills::SHARED_SKILLS mirror).",
        )


class PlannedCopiesTests(unittest.TestCase):
    def test_one_dest_per_source_file(self):
        pairs = regen.planned_skill_copies(ROOT)
        # At least one file per shared skill.
        self.assertGreaterEqual(len(pairs), len(regen.SHARED_SKILLS))
        dest_root = ROOT / "crates" / "chelis-conformance" / "assets" / "skills"
        for src, dest in pairs:
            self.assertTrue(src.is_file(), f"source missing: {src}")
            self.assertTrue(
                str(dest).startswith(str(dest_root)),
                f"dest escapes assets/skills/: {dest}",
            )

    def test_agents_contract_is_an_embedded_canonical_copy(self):
        pairs = regen.planned_canonical_copies(ROOT)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], ROOT / "AGENTS.md")
        self.assertTrue(
            str(pairs[0][1]).endswith("assets/canonical/agents-inheritance.md")
        )

    def test_every_skill_is_copied_to_both_agent_surfaces(self):
        pairs = regen.planned_agent_surface_copies(ROOT)
        self.assertEqual(len(pairs), 2 * len(regen.SHARED_SKILLS))
        destinations = {dest for _, dest in pairs}
        for skill in regen.SHARED_SKILLS:
            self.assertIn(ROOT / ".claude/skills" / skill / "SKILL.md", destinations)
            self.assertIn(ROOT / ".codex/skills" / skill / "SKILL.md", destinations)

    def test_agent_surfaces_must_be_real_directories(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "agent-skills").mkdir()
            (root / ".claude").mkdir()
            (root / ".codex").mkdir()
            (root / ".claude/skills").symlink_to("../agent-skills", target_is_directory=True)
            (root / ".codex/skills").mkdir()
            reasons = regen.agent_surface_layout_reasons(root)
            self.assertEqual(len(reasons), 1)
            self.assertIn("symlink", reasons[0])


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
        pairs = (
            regen.planned_skill_copies(ROOT)
            + regen.planned_agent_surface_copies(ROOT)
            + regen.planned_canonical_copies(ROOT)
        )
        dest_root = ROOT / "crates" / "chelis-conformance" / "assets" / "skills"
        self.assertEqual(
            regen.is_stale(pairs, dest_root),
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
            saved = regen.SHARED_SKILLS
            regen.SHARED_SKILLS = ["spec-sync"]
            try:
                with self.assertRaises(SystemExit):
                    regen.planned_skill_copies(root)
            finally:
                regen.SHARED_SKILLS = saved

    def test_single_file_skill_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._one_skill_root(Path(td), "SKILL.md")
            saved = regen.SHARED_SKILLS
            regen.SHARED_SKILLS = ["spec-sync"]
            try:
                pairs = regen.planned_skill_copies(root)
                self.assertEqual(len(pairs), 1)
                self.assertTrue(str(pairs[0][0]).endswith("spec-sync/SKILL.md"))
            finally:
                regen.SHARED_SKILLS = saved


if __name__ == "__main__":
    unittest.main()
