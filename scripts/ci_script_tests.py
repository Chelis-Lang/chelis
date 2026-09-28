#!/usr/bin/env python3
"""Disjoint hosted Python selections, with test/setup/subprocess timing receipts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

import ci_timing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# These classes execute real compiler tools, not just Python logic. Adding a
# method inherits its class's cadence; removing/renaming a class fails selection.
NIGHTLY_CLASSES = frozenset({
    "test_capacity_census_graph.ActualRustdoc",
    "test_capacity_census_wire_adapters.ConstGenericGraph",
    "test_capacity_census_wire_adapters.ActualCanonicalCodec",
    "test_capacity_census_wire_calls.CargoOriginControls",
    "test_capacity_census_wire_calls.DriverBuildControls",
    "test_capacity_census_wire_calls.InvocationControls",
    "test_capacity_census_wire_local.LocalPublicationControls",
    "test_capacity_census_wire_publication.ActualPublicationArtifacts",
    "test_capacity_census_wire_schema.ActualSchemaCodec",
    "test_capacity_census_native_calls.NativeCompilerCollectionControls",
    "test_capacity_census_native_execution.NativeExecutionIntegration",
    "test_capacity_census_native_owners.NativeOwnerIntegration",
    "test_regenerate_chelis_std_bundle.RealGeneratorFixedPointTests",
})
NATIVE_EXECUTION_TARGET = ROOT / "target/agents/native-execution-integration"
# Dedicated workflow/profile owners execute these outside the broad script lanes.
PROFILE_CLASSES = frozenset({
    "test_nextest_profile_partition.ProfilePartitionTests",
    "test_nextest_profile_partition.GeneralizationPartitionTests",
    "test_nix_flake_contract.NixFlakeContractTests",
})


def census_controls():
    from capacity_census_wire_acceptance import MUTATION_CONTROLS
    from capacity_census_compiler_json import PYTHON_CASES
    return set(MUTATION_CONTROLS) | set(PYTHON_CASES)


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def partition(suite):
    tests = list(flatten(suite))
    names = [t.id() for t in tests]
    if len(names) != len(set(names)):
        raise ValueError("duplicate discovered test identity")
    if any(isinstance(t, unittest.loader._FailedTest) for t in tests):
        raise ValueError("test discovery failed")
    classes = {name.rsplit(".", 1)[0] for name in names}
    missing = (NIGHTLY_CLASSES | PROFILE_CLASSES) - classes
    if missing:
        raise ValueError(f"missing nightly classes: {sorted(missing)}")
    owners = census_controls()
    if owners - set(names):
        raise ValueError(f"missing census controls: {sorted(owners - set(names))}")
    groups = {"pr": [], "nightly": [], "census": [], "profiles": []}
    for test in tests:
        name = test.id()
        cls = name.rsplit(".", 1)[0]
        owner = "pr"
        if cls in PROFILE_CLASSES:
            owner = "profiles"
        elif cls in NIGHTLY_CLASSES:
            owner = "census" if name in owners else "nightly"
        groups[owner].append(test)
    if not all(groups.values()):
        raise ValueError("empty Python execution owner")
    return groups


class TimedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.started = time.monotonic()
        self.issue_count = len(self.failures) + len(self.errors)
        self.skip_count = len(self.skipped)
        ci_timing.record({"event": "start", "kind": "test", "name": test.id()})

    def stopTest(self, test):
        ci_timing.record({"event": "finish", "kind": "test", "name": test.id(),
                          "seconds": time.monotonic() - self.started,
                          "outcome": "failure" if len(self.failures) + len(self.errors) > self.issue_count
                          else "skipped" if len(self.skipped) > self.skip_count else "success"})
        super().stopTest(test)


class TimedSuite(unittest.TestSuite):
    def _handleClassSetUp(self, test, result):
        if test.__class__ != getattr(result, "_previousTestClass", None):
            with ci_timing.span(test.id().rsplit(".", 1)[0] + ".setUpClass", "setup") as timing:
                errors, skips = len(result.errors), len(result.skipped)
                value = super()._handleClassSetUp(test, result)
                if len(result.errors) > errors:
                    timing["outcome"] = "failure"
                elif len(result.skipped) > skips:
                    timing["outcome"] = "skipped"
                return value


def execute(tests, stream=None):
    return unittest.TextTestRunner(resultclass=TimedResult, stream=stream).run(TimedSuite(tests))


def passed(result, selection):
    return (result.wasSuccessful() and result.testsRun > 0
            and (selection == "pr" or not result.skipped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", choices=("pr", "nightly"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    groups = partition(unittest.defaultTestLoader.discover(str(ROOT / "scripts")))
    receipt = {owner: [t.id() for t in tests] for owner, tests in groups.items()}
    output = ROOT / "target/ci-script-tests" / args.selection
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({owner: len(tests) for owner, tests in groups.items()}), flush=True)
    if args.list:
        return 0
    os.environ.setdefault("CHELIS_CI_TIMING_DIR", str(output / "timings"))
    if args.selection == "nightly":
        os.environ.setdefault(
            "CHELIS_NATIVE_EXECUTION_TARGET",
            str(NATIVE_EXECUTION_TARGET),
        )
    with ci_timing.subprocesses():
        result = execute(groups[args.selection])
    return 0 if passed(result, args.selection) else 1


if __name__ == "__main__":
    sys.exit(main())
