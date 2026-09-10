"""Lock daily ownership of exhaustive CI without running a Rust build."""
import copy
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
MOVED = {
    "dtype-phase3-oracle": ".venv/bin/python scripts/dtype_phase3_oracle.py",
    "faithful-observation-phase2-oracle": ".venv/bin/python scripts/faithful_observation_phase2_oracle.py",
    "compiled-value-ownership-phase0-oracle": ".venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2",
    "runtime-representation-phase0-oracle": "python3 scripts/gate.py runtime-representation",
    "generalize-sweep-oracle-shard": "cargo nextest run --workspace --profile ci-full --ignore-default-filter --features chelis-types/generalize-sweep-oracle",
}


def assert_extended(test, pr, nightly):
    test.assertEqual(nightly[True], {"schedule": [{"cron": "17 3 * * *"}], "workflow_dispatch": None})
    jobs = nightly["jobs"]
    for name, command in MOVED.items():
        test.assertNotIn(name, pr["jobs"])
        job = jobs[name]
        test.assertNotIn("if", job)
        test.assertFalse(job.get("continue-on-error", False))
        test.assertEqual(job["timeout-minutes"], 60 if name.startswith("generalize") else 45)
        steps = [s for s in job["steps"] if s.get("run", "").startswith(command)]
        test.assertEqual(len(steps), 1, command)
        test.assertNotIn("if", steps[0])
        test.assertFalse(steps[0].get("continue-on-error", False))
    for name in ("full-workspace", "integration-support", "backend-sanitizers-full"):
        job = jobs[name]
        test.assertNotIn("if", job)
        test.assertFalse(job.get("continue-on-error", False))
        for step in job["steps"]:
            if step.get("run", "").startswith(("cargo ", "python3 scripts/gate.py")):
                test.assertFalse(step.get("continue-on-error", False))
                if "--ignored" in step["run"]:
                    test.assertEqual(step.get("if"), "always()")
                else:
                    test.assertNotIn("if", step)
    full = jobs["full-workspace"]
    test.assertEqual(full["timeout-minutes"], 60)
    capacity = lambda job: [s for s in job["steps"] if s.get("name") == "Restore capacity rustdoc build"]
    test.assertEqual(len(capacity(full)), 1)
    test.assertEqual(capacity(full), capacity(jobs["dtype-phase3-oracle"]))
    test.assertNotIn("strategy", full)
    commands = [s.get("run") for s in full["steps"]]
    test.assertIn("cargo build --workspace --lib --bins", commands)
    test.assertIn("cargo nextest run --workspace --profile ci-full --ignore-default-filter --no-fail-fast", commands)
    test.assertIn("cargo test -p chelis-cli --test chelis_std_self_test_corpus -- --ignored --nocapture", commands)
    test.assertIn("cargo test -p chelis-backend-c", [s.get("run") for s in jobs["backend-sanitizers-full"]["steps"]])
    support = jobs["integration-support"]
    test.assertEqual(support["strategy"]["matrix"]["slice"], ["frontend", "domain"])
    test.assertIn("python3 scripts/gate.py integration --support-only --support-slice ${{ matrix.slice }}", [s.get("run") for s in support["steps"]])
    test.assertNotIn("--support-only", str(pr))
    test.assertNotIn("ProfilePartitionTests", str(pr))
    test.assertIn("ProfilePartitionTests", str(full))
    test.assertEqual(pr["jobs"]["integration"]["needs"], ["changes", "ci-fast"])
    report = jobs["report"]
    test.assertEqual(set(report["needs"]), set(MOVED) | {"full-workspace", "integration-support", "backend-sanitizers-full"})
    test.assertIn("always()", report["if"])
    test.assertEqual(report["steps"][0]["env"]["RESULTS"], "${{ toJSON(needs) }}")


class ExtendedCadenceTests(unittest.TestCase):
    def setUp(self):
        self.pr = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        self.nightly = yaml.safe_load((ROOT / ".github/workflows/heavy-e2e.yml").read_text())

    def test_extended_coverage_is_daily_and_explicit(self):
        assert_extended(self, self.pr, self.nightly)

    def test_missing_skipped_or_nonblocking_oracle_is_rejected(self):
        for name in MOVED:
            for mutation in ("remove", "skip", "ignore", "command"):
                with self.subTest(job=name, mutation=mutation):
                    nightly = copy.deepcopy(self.nightly)
                    if mutation == "remove":
                        del nightly["jobs"][name]
                    elif mutation == "skip":
                        nightly["jobs"][name]["if"] = "false"
                    elif mutation == "ignore":
                        nightly["jobs"][name]["continue-on-error"] = True
                    else:
                        nightly["jobs"][name]["steps"] = []
                    with self.assertRaises((AssertionError, KeyError)):
                        assert_extended(self, self.pr, nightly)

    def test_skipped_full_or_support_or_sanitizer_execution_is_rejected(self):
        for name in ("full-workspace", "integration-support", "backend-sanitizers-full"):
            nightly = copy.deepcopy(self.nightly)
            nightly["jobs"][name]["if"] = "false"
            with self.subTest(job=name), self.assertRaises(AssertionError):
                assert_extended(self, self.pr, nightly)

    def test_filtered_or_sharded_full_workspace_is_rejected(self):
        for mutation in ("filter", "shard"):
            nightly = copy.deepcopy(self.nightly)
            job = nightly["jobs"]["full-workspace"]
            if mutation == "shard":
                job["strategy"] = {"matrix": {"shard": [1, 2]}}
            else:
                for step in job["steps"]:
                    if step.get("run", "").startswith("cargo nextest run"):
                        step["run"] = step["run"].replace(" --ignore-default-filter", "")
            with self.assertRaises(AssertionError):
                assert_extended(self, self.pr, nightly)

    def test_missing_or_divergent_full_capacity_cache_is_rejected(self):
        for mutation in ("remove", "key"):
            nightly = copy.deepcopy(self.nightly)
            steps = nightly["jobs"]["full-workspace"]["steps"]
            cache = next(s for s in steps if s.get("name") == "Restore capacity rustdoc build")
            if mutation == "remove":
                steps.remove(cache)
            else:
                cache["with"]["key"] += "-wrong"
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                assert_extended(self, self.pr, nightly)


if __name__ == "__main__":
    unittest.main()
