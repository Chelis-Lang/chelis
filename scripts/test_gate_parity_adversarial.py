"""Mutation coverage for hosted job-scope and cache-writer policy boundaries."""

import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CACHE_JOB_ANCHOR = "\n  ci-fast:"


def _load_test_gate():
    """Load `scripts/test_gate.py` as a module so we can drive its
    `CiParityTests` against a mutated workflow file."""
    spec = importlib.util.spec_from_file_location(
        "test_gate", REPO_ROOT / "scripts" / "test_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["test_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_parity_against(workflow_text: str) -> unittest.TestResult:
    """Write `workflow_text` to a temp file, point `test_gate.CI_YML` at
    it, and run the whole `CiParityTests` suite. Returns the result."""
    tg = _load_test_gate()
    with tempfile.TemporaryDirectory() as tmp:
        # The parity suite also checks explicitly owned nightly jobs. Retain
        # sibling workflows while mutating only the ordinary CI document.
        for sibling in CI_YML.parent.glob("*.yml"):
            if sibling.name != "ci.yml":
                shutil.copyfile(sibling, Path(tmp) / sibling.name)
        path = Path(tmp) / "ci.yml"
        path.write_text(workflow_text)
        tg.CI_YML = path
        suite = unittest.TestLoader().loadTestsFromTestCase(tg.CiParityTests)
        return unittest.TextTestRunner(
            stream=io.StringIO(), verbosity=0
        ).run(suite)


class GateParityAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")
        self.ci_text = CI_YML.read_text()

    def test_control_unmutated_workflow_passes_parity(self):
        # Sanity floor: the real workflow must pass the lock, otherwise
        # the mutation tests below prove nothing.
        result = _run_parity_against(self.ci_text)
        self.assertEqual(
            (len(result.failures), len(result.errors)),
            (0, 0),
            "the unmutated ci.yml fails the parity lock; the lock is "
            "already broken independent of any mutation",
        )


    def test_cache_census_rejects_every_valid_hidden_writer_spelling(self):
        with_maps = (
            '        with:\n          shared-key: "linux-workspace"\n'
            "          save-if: true\n",
            "        with:\n          shared-key: 'linux-workspace'\n"
            "          save-if: true\n",
            "        with:\n          shared-key: &workspace_key linux-workspace\n"
            "          save-if: true\n",
            "        with:\n          shared-key: !!str linux-workspace\n"
            "          save-if: true\n",
            "        with:\n          shared-key: >-\n"
            "            linux-workspace\n          save-if: true\n",
            '        with:\n          "shared-key": linux-workspace\n'
            "          save-if: true\n",
            "        with:\n          ? shared-key\n"
            "          : linux-workspace\n          save-if: true\n",
            "        with: {shared-key: linux-workspace, save-if: true}\n",
            "        with:\n          shared-key: linux-workspace # writer\n"
            "          save-if: true\n",
            "        with:\n          shared-key : linux-workspace\n"
            "          save-if: true\n",
            "        with:\n          shared-key: ${{ 'linux-workspace' }}\n"
            "          save-if: true\n",
            "        with:\n          shared-key: "
            "${{ format('linux-{0}', 'workspace') }}\n"
            "          save-if: true\n",
            "        with:\n          shared-key: linux-${{ 'workspace' }}\n"
            "          save-if: true\n",
        )
        for with_map in with_maps:
            with self.subTest(with_map=with_map):
                step = (
                    "\n      - name: Hidden competing cache writer\n"
                    "        uses: Swatinem/rust-cache@v2\n"
                    + with_map
                )
                mutated = self.ci_text.replace(
                    CACHE_JOB_ANCHOR,
                    step + CACHE_JOB_ANCHOR,
                    1,
                )
                self.assertNotEqual(
                    mutated, self.ci_text, "mutation did not apply"
                )
                result = _run_parity_against(mutated)
                self.assertGreater(
                    len(result.failures) + len(result.errors),
                    0,
                    "the workflow-wide cache census accepted a hidden writer",
                )

    def test_underscore_job_id_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            "\n  Unclassified_job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored a valid job id containing underscore "
            "and uppercase characters",
        )

    def test_quoted_job_id_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            '\n  "Quoted_Job":\n'
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored a valid quoted job id",
        )

    def test_anchored_job_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            "\n  Hidden_Job: &hidden_job\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored an anchored job",
        )



if __name__ == "__main__":
    unittest.main()
