"""Lock daily ownership of exhaustive CI without running a Rust build."""
import copy
import re
from pathlib import Path
import tomllib
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
NEXTEST_CONFIG = ROOT / ".config/nextest.toml"
# A `cargo nextest` command naming its profile literally, anywhere a hosted job
# can reach: the workflow files themselves, and the CI-invoked Python drivers.
PROFILE_CALL = re.compile(r"--profile[\"',\s=]+([a-z][a-z0-9-]*)")
# Named by `.config/nextest.toml` as the unattended heavy-e2e selection. No
# workflow invokes it by that literal today; requiring the backstop anyway
# means a job that starts using it cannot arrive without one.
ALWAYS_UNATTENDED = {"nightly"}
MOVED = {
    "dtype-phase3-oracle": "python scripts/dtype_phase3_oracle.py",
    "faithful-observation-phase2-oracle": "python scripts/faithful_observation_phase2_oracle.py",
    "compiled-value-ownership-phase0-oracle": "python scripts/compiled_value_ownership_oracle.py --phase 2",
    "runtime-representation-phase0-oracle": "chelis-gate runtime-representation",
    "generalize-sweep-oracle-shard": "cargo nextest run --workspace --profile ci-full --ignore-default-filter --features chelis-types/generalize-sweep-oracle",
}


def assert_complete_hash_partition(test, job, command):
    """Require the shard matrix to cover the command's hash partition once.

    A nextest `--partition hash:N/M` hides nothing: every listed test lands
    in exactly one partition, so the union of shards 1..M is the whole
    selection. That only holds while the matrix really enumerates 1..M, and
    that is what this checks. It deliberately does not care what M is.
    """
    shards = job["strategy"]["matrix"]["shard"]
    partition = re.search(
        r"--partition hash:\$\{\{ matrix\.shard \}\}/(\d+)", command
    )
    test.assertIsNotNone(partition, "sharded job without a hash partition")
    count = int(partition.group(1))
    test.assertEqual(
        shards,
        list(range(1, count + 1)),
        "the shard matrix must cover every nextest hash partition exactly once",
    )


def assert_extended(test, pr, nightly):
    test.assertEqual(nightly[True], {"schedule": [{"cron": "17 3 * * *"}], "workflow_dispatch": None})
    jobs = nightly["jobs"]
    for name, command in MOVED.items():
        test.assertNotIn(name, pr["jobs"])
        job = jobs[name]
        test.assertNotIn("if", job)
        test.assertFalse(job.get("continue-on-error", False))
        execution_budget = (
            60
            if name.startswith("generalize")
            else 45
        )
        if name in ("runtime-representation-phase0-oracle", "dtype-phase3-oracle"):
            # chelis#2394: a hosted candidate dispatch builds the census cold
            # under observation, so both census-running oracles may use the
            # six-hour GitHub-hosted maximum.
            test.assertEqual(job["timeout-minutes"], 360)
        else:
            # Keep the complete oracle budget as well as cold Devenv setup headroom.
            test.assertEqual(job["timeout-minutes"], execution_budget + 25)
        if name == "runtime-representation-phase0-oracle":
            test.assertEqual(job["name"], "Runtime Representation Phase 2 Oracle")
            step_names = [step.get("name") for step in job["steps"]]
            test.assertEqual(
                [
                    step.get("run")
                    for step in job["steps"]
                    if step.get("name") == "Install Python binding dependencies"
                ],
                [
                    'uv pip install --python "$PYO3_PYTHON" '
                    "-r bindings/python/pyproject.toml"
                ],
            )
            test.assertLess(
                step_names.index("Install Python binding dependencies"),
                step_names.index("Gate (runtime representation stage)"),
            )
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
            if step.get("run", "").startswith(("cargo ", "chelis-gate")):
                test.assertFalse(step.get("continue-on-error", False))
                if "--ignored" in step["run"]:
                    # `always()` keeps a manual gate running after an earlier
                    # step failed. On a hash-partitioned job it is not part of
                    # the partition, so it also names the one shard that owns
                    # it rather than repeating on every shard. Keyed on the
                    # shard matrix specifically: a job matrixed on something
                    # else, as `integration-support` is on `slice`, has no
                    # `matrix.shard`, and demanding that condition of it would
                    # force a future `--ignored` step onto a test that is never
                    # true, so the step would never run.
                    sharded = "shard" in job.get("strategy", {}).get("matrix", {})
                    test.assertEqual(
                        step.get("if"),
                        "always() && matrix.shard == 1" if sharded else "always()",
                    )
                else:
                    test.assertNotIn("if", step)
    full = jobs["full-workspace"]
    capacity = lambda job: [s for s in job["steps"] if s.get("name") == "Restore capacity rustdoc build"]
    test.assertFalse(capacity(full))
    test.assertFalse(capacity(jobs["generalize-sweep-oracle-shard"]))
    test.assertEqual(len(capacity(jobs["dtype-phase3-oracle"])), 1)
    # chelis#1819: one unsharded run of this selection has never finished
    # inside any budget, so it never reported a verdict at all. The selection
    # is unchanged and still unfiltered; it executes as four disjoint hash
    # partitions of itself.
    workspace_suite = "cargo nextest run --workspace --profile ci-full --ignore-default-filter --no-fail-fast -E 'not (binary_id(/^chelis-compiler-api::capacity_census_wire$/) | binary_id(/^chelis-python::capacity_census_bindings$/))' --partition hash:${{ matrix.shard }}/4"
    commands = [s.get("run") for s in full["steps"]]
    test.assertIn("cargo build --workspace --lib --bins", commands)
    test.assertIn(workspace_suite, commands)
    assert_complete_hash_partition(test, full, workspace_suite)
    test.assertIn("python scripts/ci_script_tests.py nightly", [s.get("run") for s in jobs["script-nightly"]["steps"]])
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
        cached_paths = step["with"]["path"].splitlines()
        test.assertIn("target/agents/native-execution-integration", cached_paths)
        test.assertIn("target/agents/native-owner-integration", cached_paths)
    test.assertIn("cargo test -p chelis-cli --test chelis_std_self_test_corpus -- --ignored --nocapture", commands)
    test.assertIn("cargo test -p chelis-backend-c", [s.get("run") for s in jobs["backend-sanitizers-full"]["steps"]])
    support = jobs["integration-support"]
    test.assertEqual(support["strategy"]["matrix"]["slice"], ["frontend", "domain"])
    test.assertIn("chelis-gate integration --support-only --support-slice ${{ matrix.slice }}", [s.get("run") for s in support["steps"]])
    test.assertNotIn("--support-only", str(pr))
    test.assertNotIn("ProfilePartitionTests", str(pr))
    test.assertIn("ProfilePartitionTests", str(full))
    test.assertEqual(
        pr["jobs"]["integration"]["needs"],
        ["changes", "ci-fast", "change-owned-report"],
    )
    # chelis#1742: the runtime-extent oracle runs in this workflow and
    # nowhere else, so the nightly is the only place its receipts are
    # enforced. Since the B2b-3 flip that is ONE command, `--phase final`,
    # chelis#1277's completion oracle: it runs both registered phases'
    # deduplicated targets, checks the same receipts, digests and lattice,
    # and refuses `--allow-shortfall` outright. The equality is the point.
    # A second step that named a phase which CAN be excused would let the
    # job report a shortfall again, and that is what this rejects.
    extents = jobs["runtime-extent-oracle"]
    test.assertNotIn("runtime-extent-oracle", pr["jobs"])
    test.assertNotIn("if", extents)
    test.assertFalse(extents.get("continue-on-error", False))
    test.assertEqual(extents["timeout-minutes"], 115)
    extent_commands = [step.get("run") or "" for step in extents["steps"]]
    test.assertEqual(
        [
            command
            for command in extent_commands
            if "runtime_extent_oracle.py" in command
        ],
        ["python scripts/runtime_extent_oracle.py --phase final"],
    )
    # The equality already rejects every spelling that names the oracle
    # script, appended flag included, and the negative twin below measures
    # both. This says the narrower thing the equality cannot: no step of this
    # job mentions the flag at all, so a wrapper or an interpolated argument
    # that reached it without naming the script is refused too.
    test.assertFalse(
        any("--allow-shortfall" in command for command in extent_commands),
        "the nightly extent oracle may not excuse a row shortfall",
    )
    for step in extents["steps"]:
        if "runtime_extent_oracle.py" in step.get("run", ""):
            test.assertNotIn("if", step)
        test.assertFalse(step.get("continue-on-error", False))
    report = jobs["report"]
    test.assertEqual(
        set(report["needs"]),
        set(MOVED)
        | {
            "full-workspace",
            "script-nightly",
            "integration-support",
            "backend-sanitizers-full",
            "runtime-extent-oracle",
        },
    )
    test.assertIn("always()", report["if"])
    test.assertEqual(report["steps"][0]["env"]["RESULTS"], "${{ toJSON(needs) }}")


def unattended_nextest_profiles():
    """Every nextest profile a hosted job names in a LITERAL `--profile`.

    That is what this set is, and it is narrower than "every profile an
    unattended job uses". `runtime-representation` and `builtin-atom-closure`
    are selected through a variable, so the scan cannot see them, and
    `runtime-representation` is in fact run unattended by heavy-e2e's
    `runtime-representation-phase0-oracle` job, whose display name chelis#1828
    moved to "Runtime Representation Phase 2 Oracle"; the job id is cited here
    because the display name has now moved twice.
    They stay out because both ALSO run on a developer workstation,
    which is the chelis#1607 contention case that keeps `default` uncapped; a
    profile-wide cap would reach those local runs too. That is the whole
    reason. They are not excluded for owning their own failure semantics --
    `runtime_representation_phase1.py` passes no `timeout=` to `subprocess.run`,
    so an endless row there reproduces chelis#1831 rather than being contained.
    Residual recorded on chelis#1829.
    """
    found = set(ALWAYS_UNATTENDED)
    sources = sorted((ROOT / ".github/workflows").glob("*.yml"))
    sources += sorted((ROOT / "scripts").glob("*.py"))
    for source in sources:
        if source.name.startswith("test_"):
            continue
        text = source.read_text()
        if source.suffix == ".yml":
            # A toolchain-install step's `rustup --profile` is not a nextest
            # selection merely because another step in its workflow runs tests.
            commands = [
                step.get("run", "")
                for job in yaml.safe_load(text).get("jobs", {}).values()
                for step in job.get("steps", [])
            ]
        else:
            # Python drivers spell the flag and value on separate list elements.
            commands = [text]
        for command in commands:
            if "nextest" in command:
                found.update(PROFILE_CALL.findall(command))
    return found


def resolved_slow_timeout(profiles, name):
    """`name`'s effective slow-timeout, following `inherits` then `default`."""
    seen = set()
    while name and name not in seen:
        seen.add(name)
        profile = profiles.get(name, {})
        if "slow-timeout" in profile:
            return profile["slow-timeout"]
        name = profile.get("inherits", "default" if name != "default" else None)
    return None


class ExtendedCadenceTests(unittest.TestCase):
    def setUp(self):
        self.pr = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        self.nightly = yaml.safe_load((ROOT / ".github/workflows/heavy-e2e.yml").read_text())

    def test_extended_coverage_is_daily_and_explicit(self):
        assert_extended(self, self.pr, self.nightly)

    def test_missing_skipped_or_nonblocking_oracle_is_rejected(self):
        for name in MOVED:
            for mutation in ("remove", "skip", "ignore", "command", "timeout"):
                with self.subTest(job=name, mutation=mutation):
                    nightly = copy.deepcopy(self.nightly)
                    if mutation == "remove":
                        del nightly["jobs"][name]
                    elif mutation == "skip":
                        nightly["jobs"][name]["if"] = "false"
                    elif mutation == "ignore":
                        nightly["jobs"][name]["continue-on-error"] = True
                    elif mutation == "command":
                        nightly["jobs"][name]["steps"] = []
                    else:
                        nightly["jobs"][name]["timeout-minutes"] = 1
                    with self.assertRaises((AssertionError, KeyError)):
                        assert_extended(self, self.pr, nightly)

    def test_a_skipped_or_nonblocking_runtime_extent_oracle_is_rejected(self):
        # Negative parity for the block above: every way the extent oracle
        # could stop enforcing must fail this lock, including dropping it
        # from the nightly report's needs, which is what would let it go red
        # unnoticed.
        for mutation in (
            "remove",
            "skip",
            "ignore",
            "steps",
            "needs",
            "downgrade",
            "shortfall",
            "timeout",
        ):
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
            elif mutation == "downgrade":
                # The completion oracle swapped for a phase whose row
                # shortfall can be excused: the job would still be green,
                # still be named, and no longer enforce the exit.
                job["steps"][-1]["run"] = job["steps"][-1]["run"].replace(
                    "--phase final", "--phase b --allow-shortfall"
                )
            elif mutation == "timeout":
                job["timeout-minutes"] = 45
            else:
                # The subtler half: the command the equality accepts, with
                # the flag appended. `--phase final` refuses the flag at run
                # time, so this one would fail the nightly loudly rather than
                # excuse anything, but a guard that let it through would also
                # let through the phase spelling that does excuse a
                # shortfall. Rejecting it here keeps both out.
                job["steps"][-1]["run"] += " --allow-shortfall"
            with self.subTest(mutation=mutation):
                with self.assertRaises((AssertionError, KeyError)):
                    assert_extended(self, self.pr, nightly)

    def test_skipped_full_or_support_or_sanitizer_execution_is_rejected(self):
        for name in ("full-workspace", "script-nightly", "integration-support", "backend-sanitizers-full"):
            nightly = copy.deepcopy(self.nightly)
            nightly["jobs"][name]["if"] = "false"
            with self.subTest(job=name), self.assertRaises(AssertionError):
                assert_extended(self, self.pr, nightly)

    def test_filtered_or_incompletely_partitioned_full_workspace_is_rejected(self):
        # What this guard protects is that no default filter can hide a newly
        # added target from the nightly, and that everything the selection
        # lists still executes. A `--ignore-default-filter` drop breaks the
        # first; an incomplete `--partition hash:N/M` matrix breaks the second.
        # A complete partition breaks neither -- every listed test lands in
        # exactly one shard -- so sharding itself is no longer the mutation
        # under test. chelis#1819.
        def nextest_step(job):
            return next(
                step
                for step in job["steps"]
                if step.get("run", "").startswith("cargo nextest run")
            )

        for mutation in (
            "filter",
            "missing-shard",
            "extra-shard",
            "partition-count",
            "partition-dropped",
            "matrix-dropped",
        ):
            nightly = copy.deepcopy(self.nightly)
            job = nightly["jobs"]["full-workspace"]
            if mutation == "filter":
                step = nextest_step(job)
                step["run"] = step["run"].replace(" --ignore-default-filter", "")
            elif mutation == "missing-shard":
                job["strategy"]["matrix"]["shard"] = [1, 2, 3]
            elif mutation == "extra-shard":
                job["strategy"]["matrix"]["shard"] = [1, 2, 3, 4, 5]
            elif mutation == "partition-count":
                step = nextest_step(job)
                step["run"] = step["run"].replace(
                    "matrix.shard }}/4", "matrix.shard }}/2"
                )
            elif mutation == "partition-dropped":
                step = nextest_step(job)
                step["run"] = step["run"].split(" --partition ")[0]
            else:
                del job["strategy"]
            with self.subTest(mutation=mutation), self.assertRaises(
                (AssertionError, KeyError)
            ):
                assert_extended(self, self.pr, nightly)

    def test_a_complete_hash_partition_is_accepted_at_any_width(self):
        # Positive parity for the narrowing above: the guard must not reject a
        # partition merely for being one. Any width is a complete partition as
        # long as the matrix enumerates it; the reviewed width itself is pinned
        # by the verbatim command in `assert_extended`.
        for width in (2, 4, 6):
            with self.subTest(width=width):
                assert_complete_hash_partition(
                    self,
                    {"strategy": {"matrix": {"shard": list(range(1, width + 1))}}},
                    "cargo nextest run --workspace "
                    f"--partition hash:${{{{ matrix.shard }}}}/{width}",
                )

    def test_census_cannot_silently_return_to_full_workspace(self):
        nightly = copy.deepcopy(self.nightly)
        for step in nightly["jobs"]["full-workspace"]["steps"]:
            if step.get("run", "").startswith("cargo nextest run"):
                step["run"] = step["run"].split(" -E ")[0]
        with self.assertRaises(AssertionError):
            assert_extended(self, self.pr, nightly)


class UnattendedBackstopTests(unittest.TestCase):
    """chelis#1831: an unattended profile without `terminate-after` lets one
    pathological row consume the whole hosted budget and report nothing."""

    def setUp(self):
        self.config = tomllib.loads(NEXTEST_CONFIG.read_text())

    def test_the_scan_finds_the_profiles_hosted_jobs_actually_run(self):
        # A scan that silently found nothing would pass every other test here.
        found = unattended_nextest_profiles()
        self.assertLessEqual({"ci", "ci-fast", "ci-full", "nightly"}, found)
        for name in found:
            self.assertIn(name, self.config["profile"], name)

    def test_every_unattended_profile_terminates_a_pathological_row(self):
        profiles = self.config["profile"]
        for name in sorted(unattended_nextest_profiles()):
            with self.subTest(profile=name):
                timeout = resolved_slow_timeout(profiles, name)
                self.assertIsNotNone(
                    timeout,
                    f"profile `{name}` runs unattended and sets no slow-timeout",
                )
                self.assertIn(
                    "terminate-after",
                    timeout,
                    f"profile `{name}` warns forever and never kills the test",
                )
                self.assertGreater(timeout["terminate-after"], 0)

    def test_a_profile_whose_backstop_is_removed_is_rejected(self):
        for mutation in ("drop-timeout", "drop-terminate-after"):
            profiles = copy.deepcopy(self.config["profile"])
            if mutation == "drop-timeout":
                profiles["ci-full"].pop("slow-timeout")
            else:
                profiles["ci-full"]["slow-timeout"].pop("terminate-after")
            with self.subTest(mutation=mutation):
                timeout = resolved_slow_timeout(profiles, "ci-full")
                self.assertTrue(timeout is None or "terminate-after" not in timeout)

    def test_the_local_default_profile_keeps_no_backstop(self):
        # chelis#1607: a load-sensitive cap on a contended workstation fires as
        # a phantom failure. Inheriting one into `default` would put every
        # local `cargo nextest run` under it.
        self.assertNotIn("slow-timeout", self.config["profile"]["default"])


if __name__ == "__main__":
    unittest.main()
