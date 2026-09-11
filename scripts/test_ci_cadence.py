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
        test.assertEqual(job["timeout-minutes"], 60 if name.startswith(("generalize", "dtype")) else 45)
        if name == "runtime-representation-phase0-oracle":
            test.assertEqual(job["name"], "Runtime Representation Phase 2 Oracle")
            artifacts = [
                step
                for step in job["steps"]
                if step.get("name")
                == "Upload runtime representation execution receipts"
            ]
            test.assertEqual(len(artifacts), 1)
            test.assertEqual(
                artifacts[0]["with"]["path"],
                (
                    "target/runtime-representation-phase1/\n"
                    "target/runtime-representation-phase2/\n"
                ),
            )
        steps = [s for s in job["steps"] if s.get("run", "").startswith(command)]
        test.assertEqual(len(steps), 1, command)
        test.assertNotIn("if", steps[0])
        test.assertFalse(steps[0].get("continue-on-error", False))
    for name in ("full-workspace", "script-nightly", "integration-support", "backend-sanitizers-full"):
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
    test.assertFalse(capacity(full))
    test.assertFalse(capacity(jobs["generalize-sweep-oracle-shard"]))
    test.assertEqual(len(capacity(jobs["dtype-phase3-oracle"])), 1)
    test.assertNotIn("strategy", full)
    commands = [s.get("run") for s in full["steps"]]
    test.assertIn("cargo build --workspace --lib --bins", commands)
    test.assertIn("cargo nextest run --workspace --profile ci-full --ignore-default-filter --no-fail-fast -E 'not (binary_id(/^chelis-compiler-api::capacity_census_wire$/) | binary_id(/^chelis-python::capacity_census_bindings$/))'", commands)
    test.assertIn(".venv/bin/python scripts/ci_script_tests.py nightly", [s.get("run") for s in jobs["script-nightly"]["steps"]])
    script_cache_steps = [
        step
        for step in jobs["script-nightly"]["steps"]
        if step.get("name") in {
            "Restore script compiler builds",
            "Save script compiler builds",
        }
    ]
    test.assertEqual(len(script_cache_steps), 2)
    for step in script_cache_steps:
        test.assertIn(
            "target/agents/native-owner-integration",
            step["with"]["path"].splitlines(),
        )
    test.assertIn("cargo test -p chelis-cli --test chelis_std_self_test_corpus -- --ignored --nocapture", commands)
    test.assertIn("cargo test -p chelis-backend-c", [s.get("run") for s in jobs["backend-sanitizers-full"]["steps"]])
    support = jobs["integration-support"]
    test.assertEqual(support["strategy"]["matrix"]["slice"], ["frontend", "domain"])
    test.assertIn("python3 scripts/gate.py integration --support-only --support-slice ${{ matrix.slice }}", [s.get("run") for s in support["steps"]])
    test.assertNotIn("--support-only", str(pr))
    test.assertNotIn("ProfilePartitionTests", str(pr))
    test.assertIn("ProfilePartitionTests", str(full))
    test.assertEqual(pr["jobs"]["integration"]["needs"], ["changes", "ci-fast"])
    # chelis#1742: the runtime-extent oracle runs in this workflow and
    # nowhere else, so the nightly is the only place its receipts are
    # enforced. Phase A must PASS. Phase B enforces the same receipts, the
    # same digests and the same lattice while chelis#1277's remaining rows
    # land, and `--allow-shortfall` downgrades the recorded ROW shortfall
    # alone; nothing else about the run is excused. The B2b-3 flip pull
    # request drops the flag and this assertion with it.
    extents = jobs["runtime-extent-oracle"]
    test.assertNotIn("runtime-extent-oracle", pr["jobs"])
    test.assertNotIn("if", extents)
    test.assertFalse(extents.get("continue-on-error", False))
    extent_commands = [step.get("run") for step in extents["steps"]]
    test.assertIn(
        ".venv/bin/python scripts/runtime_extent_oracle.py --phase a", extent_commands
    )
    test.assertIn(
        ".venv/bin/python scripts/runtime_extent_oracle.py --phase b --allow-shortfall",
        extent_commands,
    )
    for step in extents["steps"]:
        test.assertNotIn("if", step)
        test.assertFalse(step.get("continue-on-error", False))
    report = jobs["report"]
    test.assertEqual(set(report["needs"]), set(MOVED) | {"full-workspace", "script-nightly", "integration-support", "backend-sanitizers-full", "runtime-extent-oracle"})
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

    def test_a_skipped_or_nonblocking_runtime_extent_oracle_is_rejected(self):
        # Negative parity for the block above: every way the extent oracle
        # could stop enforcing must fail this lock, including dropping it
        # from the nightly report's needs, which is what would let it go red
        # unnoticed.
        for mutation in ("remove", "skip", "ignore", "steps", "needs", "shortfall"):
            nightly = copy.deepcopy(self.nightly)
            job = nightly["jobs"]["runtime-extent-oracle"]
            if mutation == "remove":
                del nightly["jobs"]["runtime-extent-oracle"]
            elif mutation == "skip":
                job["if"] = "false"
            elif mutation == "ignore":
                job["steps"][-1]["continue-on-error"] = True
            elif mutation == "steps":
                job["steps"] = []
            elif mutation == "needs":
                nightly["jobs"]["report"]["needs"] = [
                    name
                    for name in nightly["jobs"]["report"]["needs"]
                    if name != "runtime-extent-oracle"
                ]
            else:
                job["steps"][-1]["run"] = job["steps"][-1]["run"].replace(
                    " --allow-shortfall", " --allow-everything"
                )
            with self.subTest(mutation=mutation):
                with self.assertRaises((AssertionError, KeyError)):
                    assert_extended(self, self.pr, nightly)

    def test_skipped_full_or_support_or_sanitizer_execution_is_rejected(self):
        for name in ("full-workspace", "script-nightly", "integration-support", "backend-sanitizers-full"):
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

    def test_census_cannot_silently_return_to_full_workspace(self):
        nightly = copy.deepcopy(self.nightly)
        for step in nightly["jobs"]["full-workspace"]["steps"]:
            if step.get("run", "").startswith("cargo nextest run"):
                step["run"] = step["run"].split(" -E ")[0]
        with self.assertRaises(AssertionError):
            assert_extended(self, self.pr, nightly)


if __name__ == "__main__":
    unittest.main()
