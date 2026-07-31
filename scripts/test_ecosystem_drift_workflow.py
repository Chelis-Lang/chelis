"""Regression checks for the scheduled ecosystem drift workflow."""

from pathlib import Path
import unittest


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "ecosystem-drift.yml"
)


class EcosystemDriftWorkflowTests(unittest.TestCase):
    def test_reef_canary_raises_the_whole_suite_timeout(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        reef_test_step = text.split(
            "- name: chelis test tests/ (${{ matrix.repo }} vs HEAD)", 1
        )[1].split("# cargo legs only", 1)[0]

        self.assertIn(
            "chelis test tests/ --timeout 600 --suite-timeout 900 --jobs auto",
            reef_test_step,
        )
        self.assertEqual(reef_test_step.count("--timeout 600"), 1)
        self.assertEqual(reef_test_step.count("--suite-timeout 900"), 1)

    def test_hello_head_canary_does_not_compare_release_generated_bytes(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        hello_step = text.split(
            "- name: hello-chelis check + test + C-backend (vs HEAD)", 1
        )[1].split("# ADVISORY conformance signal", 1)[0]

        self.assertIn("run chelis check", hello_step)
        self.assertIn("run timeout 900s chelis test", hello_step)
        self.assertIn("tests/test_c_backend.py", hello_step)
        self.assertNotIn("regen_deep.py", hello_step)


if __name__ == "__main__":
    unittest.main()
