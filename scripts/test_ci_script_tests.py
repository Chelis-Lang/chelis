"""Execution ownership and timing contracts for Python CI."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_script_tests as runner
import ci_timing


class SelectionTests(unittest.TestCase):
    def test_every_discovered_test_has_one_owner_and_owned_controls_are_not_repeated(self):
        suite = unittest.defaultTestLoader.discover(str(runner.ROOT / "scripts"))
        groups = runner.partition(suite)
        identities = [{t.id() for t in group} for group in groups.values()]
        self.assertEqual(sum(map(len, identities)), len(set.union(*identities)))
        self.assertEqual(set.union(*identities), {t.id() for t in runner.flatten(suite)})
        self.assertTrue(all(identities))
        self.assertIn("test_regenerate_chelis_std_bundle.RealGeneratorFixedPointTests.test_two_real_debug_regenerations_reach_a_byte_fixed_point", identities[1])
        for name in identities[0]:
            self.assertNotIn(name.rsplit(".", 1)[0], runner.NIGHTLY_CLASSES)
        self.assertTrue(identities[2] <= runner.census_controls())

    def test_stale_class_or_duplicate_test_is_rejected(self):
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with self.assertRaisesRegex(ValueError, "missing nightly"):
            runner.partition(suite)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.partition(unittest.TestSuite([suite, suite]))


class TimingTests(unittest.TestCase):
    def test_subprocess_timing_retains_success_failure_and_exception_semantics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CHELIS_CI_TIMING_DIR": tmp}):
            original = subprocess.run
            with ci_timing.subprocesses(), ci_timing.subprocesses():
                self.assertEqual(subprocess.run([sys.executable, "-c", "pass"]).returncode, 0)
                with self.assertRaises(subprocess.CalledProcessError):
                    subprocess.run([sys.executable, "-c", "raise SystemExit(7)"], check=True)
                with self.assertRaises(FileNotFoundError):
                    subprocess.run([str(Path(tmp) / "absent")])
            self.assertIs(subprocess.run, original)
            rows = [json.loads(line) for p in Path(tmp).glob("*.jsonl") for line in p.read_text().splitlines()]
            ended = [r for r in rows if r["event"] == "finish"]
            self.assertEqual([r["outcome"] for r in ended], ["success", "CalledProcessError", "FileNotFoundError"])
            self.assertTrue(all(r["seconds"] >= 0 for r in ended))
            self.assertTrue(all("env" not in r for r in rows))

    def test_test_failures_and_class_setup_are_reported(self):
        class Example(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                pass

            def test_failure(self):
                self.fail("intentional")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CHELIS_CI_TIMING_DIR": tmp}):
            result = runner.execute(list(runner.flatten(unittest.defaultTestLoader.loadTestsFromTestCase(Example))), stream=io.StringIO())
            self.assertFalse(result.wasSuccessful())
            rows = [json.loads(line) for p in Path(tmp).glob("*.jsonl") for line in p.read_text().splitlines()]
            names = [r["name"] for r in rows]
            self.assertTrue(any(n.endswith(".setUpClass") for n in names))
            self.assertTrue(any(n.endswith(".test_failure") for n in names))

    def test_empty_or_skipped_nightly_is_not_success(self):
        result = unittest.TestResult()
        self.assertFalse(runner.passed(result, "nightly"))
        result.testsRun = 1
        self.assertTrue(runner.passed(result, "nightly"))
        result.skipped.append((unittest.FunctionTestCase(lambda: None), "missing prerequisite"))
        self.assertFalse(runner.passed(result, "nightly"))
        self.assertTrue(runner.passed(result, "pr"))


if __name__ == "__main__":
    unittest.main()
