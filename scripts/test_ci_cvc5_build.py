#!/usr/bin/env python3
"""Tests for scripts/ci_cvc5_build.py (chelis#1004).

The property under test is narrow and adversarial: the wrapper must ride out
a transient cvc5 SOURCE FETCH failure while leaving a real build failure
exactly as fatal as it is today. `docs/smt_build_setup.md` makes the
release-job cold build the authoritative proof that the shipped binary was
produced by the real recipe, so a retry that fired on any non-zero exit
would quietly convert a broken cvc5 build into a green release. Every
positive case here therefore has a negative twin.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import ci_cvc5_build

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

# Verbatim from run 30673030685, the v0.18.0 release that needed three
# attempts. Both are fetch failures at the two distinct sites in
# cvc5-sys-0.3.1/build.rs.
OBSERVED_DEP_DOWNLOAD_403 = (
    "  error: downloading "
    "'https://github.com/cvc5/cvc5-deps/blob/main/gmp-6.3.0.tar.bz2?raw=true' failed\n"
)
OBSERVED_URL_403 = "    The requested URL returned error: 403\n"
OBSERVED_CLONE_FAILURE = (
    "  thread 'main' (28443) panicked at "
    "cvc5-sys-0.3.1/build.rs:276:5:\n  git clone of cvc5 tag cvc5-1.3.1 failed\n"
)

# Real build failures. None of these may be classified as transient.
REAL_COMPILE_ERROR = "error[E0308]: mismatched types\n"
REAL_LINK_ERROR = "error: linking with `cc` failed: exit status: 1\n"
REAL_CMAKE_CONFIGURE_ERROR = (
    "CMake Error at CMakeLists.txt:42 (message):\n  Unsupported compiler\n"
)
REAL_CVC5_PANIC_WITHOUT_FETCH = (
    "thread 'main' panicked at cvc5-sys-0.3.1/build.rs:231:5: cvc5 build failed\n"
)


class FakeRunner:
    """Records invocations and replays a scripted list of outcomes."""

    def __init__(self, outcomes: list[tuple[int, bool]]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> tuple[int, bool]:
        self.calls.append(list(cmd))
        if not self.outcomes:
            raise AssertionError("runner called more times than scripted")
        return self.outcomes.pop(0)


class ClassifyLineTest(unittest.TestCase):
    def test_observed_403_download_is_transient(self) -> None:
        self.assertTrue(ci_cvc5_build.classify_line(OBSERVED_DEP_DOWNLOAD_403))

    def test_observed_url_403_is_transient(self) -> None:
        self.assertTrue(ci_cvc5_build.classify_line(OBSERVED_URL_403))

    def test_observed_clone_failure_is_transient(self) -> None:
        self.assertTrue(ci_cvc5_build.classify_line(OBSERVED_CLONE_FAILURE))

    def test_each_download_failed_is_transient(self) -> None:
        self.assertTrue(ci_cvc5_build.classify_line("    Each download failed!\n"))

    def test_classification_is_case_insensitive(self) -> None:
        self.assertTrue(
            ci_cvc5_build.classify_line("THE REQUESTED URL RETURNED ERROR: 403\n")
        )

    # Negative parity: a real failure must never be classified transient.
    def test_compile_error_is_not_transient(self) -> None:
        self.assertFalse(ci_cvc5_build.classify_line(REAL_COMPILE_ERROR))

    def test_link_error_is_not_transient(self) -> None:
        self.assertFalse(ci_cvc5_build.classify_line(REAL_LINK_ERROR))

    def test_cmake_configure_error_is_not_transient(self) -> None:
        self.assertFalse(ci_cvc5_build.classify_line(REAL_CMAKE_CONFIGURE_ERROR))

    def test_bare_cvc5_panic_is_not_transient(self) -> None:
        # build.rs:231 panics for a failed download AND for other build
        # breakage. The panic line alone must not license a retry; only the
        # download diagnostics above it do.
        self.assertFalse(ci_cvc5_build.classify_line(REAL_CVC5_PANIC_WITHOUT_FETCH))


class BuildRetryTest(unittest.TestCase):
    def _build(self, runner: FakeRunner, **kw):
        slept: list[float] = []
        rc = ci_cvc5_build.build(
            ["cargo", "build"],
            runner=runner,
            sleep=slept.append,
            backoff_seconds=1,
            **kw,
        )
        return rc, slept

    def test_success_on_first_attempt_runs_once(self) -> None:
        runner = FakeRunner([(0, False)])
        rc, slept = self._build(runner)
        self.assertEqual(rc, 0)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(slept, [], "no backoff before a first attempt")

    def test_transient_failure_then_success_retries(self) -> None:
        runner = FakeRunner([(101, True), (0, False)])
        rc, slept = self._build(runner)
        self.assertEqual(rc, 0)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(slept, [1], "exactly one backoff before the retry")

    def test_real_failure_is_not_retried(self) -> None:
        # The guard that keeps the cold-build proof honest.
        runner = FakeRunner([(101, False)])
        rc, slept = self._build(runner)
        self.assertEqual(rc, 101)
        self.assertEqual(
            len(runner.calls), 1, "a non-transient failure must fail on attempt 1"
        )
        self.assertEqual(slept, [])

    def test_real_failure_after_a_transient_one_is_not_retried(self) -> None:
        # A flaky fetch on attempt 1 must not buy extra attempts for a real
        # compile break on attempt 2.
        runner = FakeRunner([(101, True), (101, False)])
        rc, _ = self._build(runner, attempts=5)
        self.assertEqual(rc, 101)
        self.assertEqual(len(runner.calls), 2)

    def test_persistent_transient_failure_exhausts_and_fails(self) -> None:
        runner = FakeRunner([(101, True), (101, True), (101, True)])
        rc, slept = self._build(runner, attempts=3)
        self.assertEqual(rc, 101, "a sustained outage must still fail the job")
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(slept, [1, 1])

    def test_attempts_of_one_never_retries(self) -> None:
        runner = FakeRunner([(101, True)])
        rc, _ = self._build(runner, attempts=1)
        self.assertEqual(rc, 101)
        self.assertEqual(len(runner.calls), 1)

    def test_empty_command_is_a_usage_error(self) -> None:
        rc = ci_cvc5_build.build([], runner=FakeRunner([]), sleep=lambda _s: None)
        self.assertEqual(rc, 2)


class ArgParsingTest(unittest.TestCase):
    def test_command_after_double_dash_is_passed_through(self) -> None:
        args = ci_cvc5_build.parse_args(
            ["--", "cargo", "build", "--release", "-p", "chelis-cli", "--features", "smt"]
        )
        self.assertEqual(
            args.command,
            ["cargo", "build", "--release", "-p", "chelis-cli", "--features", "smt"],
        )

    def test_flags_before_the_double_dash_are_parsed(self) -> None:
        args = ci_cvc5_build.parse_args(["--attempts", "2", "--", "cargo", "build"])
        self.assertEqual(args.attempts, 2)
        self.assertEqual(args.command, ["cargo", "build"])

    def test_attempts_below_one_is_rejected(self) -> None:
        self.assertEqual(ci_cvc5_build.main(["--attempts", "0", "--", "true"]), 2)


class ReleaseWorkflowWiringTest(unittest.TestCase):
    """The wrapper is worthless unless release.yml actually calls it."""

    def setUp(self) -> None:
        self.text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    def test_every_smt_build_step_goes_through_the_wrapper(self) -> None:
        smt_builds = [
            line.strip()
            for line in self.text.splitlines()
            if re.search(r"cargo (build|rustc)\b", line) and "--features smt" in line
        ]
        self.assertEqual(
            len(smt_builds),
            5,
            "release.yml should build chelis-cli --features smt in all five jobs",
        )
        for line in smt_builds:
            self.assertIn(
                "scripts/ci_cvc5_build.py",
                line,
                "a cold cvc5 build that does not go through the retry wrapper "
                "will fail the release on a transient fetch (chelis#1004)",
            )

    def test_wrapper_line_preserves_the_cargo_target_for_the_pyo3_guard(self) -> None:
        # test_release_workflow_pyo3_isolation.py parses `cargo build -p <name>`
        # out of release.yml to decide which crates to vet. If the wrapper hid
        # the real command, that parser would find no chelis-cli and SKIP its
        # smt check instead of failing. Pin the text it depends on.
        pattern = re.compile(r"cargo\s+build(?:\s+--release)?\s+-p\s+([A-Za-z0-9_-]+)")
        self.assertIn("chelis-cli", set(pattern.findall(self.text)))

    def test_release_build_jobs_bound_their_wall_clock(self) -> None:
        # Retries extend the worst-case job, so an unbounded build job could
        # now run toward the 6-hour runner cap instead of failing. Checked
        # per job rather than by counting occurrences, so adding a ceiling
        # elsewhere in the file cannot mask a missing one here.
        for job in (
            "build-linux-x86_64:",
            "build-linux-x86_64-glibc231:",
            "build-linux-x86_64-static:",
            "build-linux-x86_64-musl:",
            "build-darwin-arm64:",
        ):
            with self.subTest(job=job):
                block = self._job_block(job)
                self.assertIn(
                    "timeout-minutes:",
                    block,
                    f"{job} has no wall-clock ceiling",
                )

    def _job_block(self, job_key: str) -> str:
        """The job's YAML from its key up to (not including) its `steps:`.

        Text-sliced rather than YAML-parsed because pyyaml is not a repo
        dependency and the sibling workflow tests parse these files as text
        too.
        """
        start = self.text.index(f"  {job_key}")
        end = self.text.index("\n    steps:", start)
        return self.text[start:end]

    def test_release_does_not_link_the_durable_prebuilt(self) -> None:
        # The invariant this change must NOT break: release.yml stays a cold
        # from-source cvc5 build (docs/smt_build_setup.md, WI-11).
        self.assertNotIn("ci_cvc5_cache.py", self.text)


if __name__ == "__main__":
    unittest.main()
