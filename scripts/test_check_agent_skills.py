"""Positive and negative controls for the shared agent-skill CI check."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import check_agent_skills as check
from scripts.regenerate_conformance_assets import SHARED_SKILLS


ROOT = Path(__file__).resolve().parents[1]


class SkillContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in SHARED_SKILLS:
            for prefix in ("agent-skills", "crates/chelis-conformance/assets/skills"):
                path = self.root / prefix / name / "SKILL.md"
                path.parent.mkdir(parents=True)
                path.write_text(f"---\nname: {name}\ndescription: Review the named contract.\n---\n\n# Instructions\n")
        for platform in (".claude", ".codex"):
            path = self.root / platform / "commands/red-team.md"
            path.parent.mkdir(parents=True)
            path.write_text("Use the shared redteam-exec skill.\n")

    def skill(self, prefix="agent-skills"):
        return self.root / prefix / "redteam-exec/SKILL.md"

    def test_current_repository_and_complete_copy_pass(self):
        self.assertEqual(check.check(ROOT), [])
        self.assertEqual(check.check(self.root), [])

    def test_yaml_block_description_and_supported_metadata_pass(self):
        path = self.skill()
        path.write_text("---\nname: redteam-exec\ndescription: >-\n  Review a change\n  against its contract.\nmetadata:\n  short-description: Review\n---\n\n# Instructions\n")
        self.assertEqual(check.validate_skill(path), [])

    def test_invalid_frontmatter_is_rejected_in_either_copy(self):
        cases = [
            "name: redteam-exec\ndescription: valid",  # missing delimiters
            "---\nname: [unterminated\n---\n",
            "---\n- a sequence\n---\n",
            "---\nname: redteam-exec\n---\n",
            "---\nname: redteam-exec\ndescription: 42\n---\n",
            "---\nname: redteam-exec\ndescription: '  '\n---\n",
            "---\nname: invalid_name\ndescription: valid\n---\n",
            "---\nname: other-skill\ndescription: valid\n---\n",
            "---\nname: redteam-exec\ndescription: valid\nextra: ignored?\n---\n",
            "---\nname: redteam-exec\nname: other\ndescription: valid\n---\n",
        ]
        for prefix in ("agent-skills", "crates/chelis-conformance/assets/skills"):
            path = self.skill(prefix)
            original = path.read_bytes()
            for text in cases:
                with self.subTest(prefix=prefix, text=text):
                    path.write_text(text)
                    self.assertTrue(check.validate_skill(path))
                    self.assertTrue(check.check(self.root))
            path.write_bytes(original)

    def test_missing_or_stale_embedded_skill_is_rejected(self):
        path = self.skill("crates/chelis-conformance/assets/skills")
        path.write_text(path.read_text() + "Outdated instructions.\n")
        self.assertTrue(check.check(self.root))
        path.unlink()
        self.assertTrue(check.check(self.root))

    def test_unfinished_instructions_are_rejected_even_when_copies_agree(self):
        paths = [self.skill(prefix) for prefix in (
            "agent-skills", "crates/chelis-conformance/assets/skills"
        )]
        original = paths[0].read_text()
        for body in (
            "[TODO: replace this unfinished instruction]\n",
            "   [TODO: finish this instruction]  \n",
            "```text\nAn example.\n```\n[TODO: finish after the example]\n",
            "~~~text\nAn example.\n~~~~\n[TODO: finish after the example]\n",
        ):
            with self.subTest(body=body):
                for path in paths:
                    path.write_text(original + body)
                    self.assertTrue(check.validate_skill(path))
                self.assertTrue(check.check(self.root))

    def test_fenced_placeholder_examples_are_valid_instructions(self):
        path = self.skill()
        original = path.read_text()
        for body in (
            "```text\n[TODO: an example placeholder]\n```\n",
            "~~~text\n[TODO: an example placeholder]\n~~~\n",
            "````text\n```\n[TODO: a shorter fence does not close]\n````\n",
            "```text\n~~~\n[TODO: a different marker does not close]\n```\n",
            "- ```text\n[TODO: a list example]\n  ```\n",
        ):
            with self.subTest(body=body):
                path.write_text(original + body)
                self.assertEqual(check.validate_skill(path), [])

    def test_missing_or_unknown_source_skill_is_rejected(self):
        path = self.skill()
        path.unlink()
        self.assertTrue(check.check(self.root))
        (self.root / "agent-skills/unregistered").mkdir()
        self.assertTrue(check.check(self.root))

    def test_missing_or_different_command_wrapper_is_rejected(self):
        path = self.root / ".codex/commands/red-team.md"
        path.write_text("Old review rules.\n")
        self.assertTrue(check.check(self.root))
        path.unlink()
        self.assertTrue(check.check(self.root))

    def test_cli_passes_real_tree_and_fails_a_stale_tree(self):
        command = [sys.executable, str(ROOT / "scripts/check_agent_skills.py"), "--root", str(self.root)]
        good = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("AGENT SKILLS: PASS", good.stdout)
        self.skill("crates/chelis-conformance/assets/skills").unlink()
        bad = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("AGENT SKILLS: FAIL", bad.stdout)


if __name__ == "__main__":
    unittest.main()
