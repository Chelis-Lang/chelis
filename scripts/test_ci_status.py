"""A CI status report must distinguish branch requirements from reporting work."""

import unittest
from unittest.mock import patch
import contextlib
import io

from scripts import ci_status


def check(name, *, ident=1, status="completed", conclusion="success", app_id=15368):
    return {"id": ident, "name": name, "status": status, "conclusion": conclusion,
            "app": {"id": app_id}, "html_url": f"https://github.com/jobs/{ident}"}


class StatusTests(unittest.TestCase):
    def report(self, checks, required=None, statuses=()):
        return ci_status.classify(
            required or [{"context": "macOS Smoke", "app_id": 15368}],
            checks, statuses,
        )

    def test_optional_queue_does_not_hide_required_success(self):
        result = self.report([check("macOS Smoke"), check("CI Test Telemetry", ident=2,
                                                        status="queued", conclusion=None)])
        self.assertTrue(result["required_passed"])
        self.assertEqual(result["other_pending"], ["CI Test Telemetry"])
        self.assertEqual(result["required_pending"], [])

    def test_required_aggregate_queue_is_pending(self):
        result = self.report([check("macOS Smoke", status="queued", conclusion=None)])
        self.assertFalse(result["required_passed"])
        self.assertEqual(result["required_pending"], ["macOS Smoke"])

    def test_missing_required_check_is_pending(self):
        result = self.report([check("CI Test Telemetry")])
        self.assertFalse(result["required_passed"])
        self.assertEqual(result["required_missing"], ["macOS Smoke"])

    def test_empty_requirements_cannot_report_success(self):
        with self.assertRaises(ValueError):
            ci_status.classify([], [], [])

    def test_wrong_app_cannot_satisfy_a_required_check(self):
        result = self.report([check("macOS Smoke", app_id=123)])
        self.assertFalse(result["required_passed"])
        self.assertEqual(result["required_missing"], ["macOS Smoke"])

    def test_latest_check_attempt_wins(self):
        result = self.report([check("macOS Smoke", ident=2, status="in_progress", conclusion=None),
                              check("macOS Smoke", ident=1)])
        self.assertFalse(result["required_passed"])

    def test_same_name_checks_from_separate_suites_must_both_pass(self):
        first = {**check("macOS Smoke"), "check_suite": {"id": 10}}
        second = {**check("macOS Smoke", ident=2, conclusion="failure"),
                  "check_suite": {"id": 20}}
        self.assertFalse(self.report([first, second])["required_passed"])

    def test_failures_and_cancelled_checks_do_not_pass(self):
        for conclusion in ("failure", "cancelled", "timed_out", "action_required", "stale", None):
            with self.subTest(conclusion=conclusion):
                result = self.report([check("macOS Smoke", conclusion=conclusion)])
                self.assertFalse(result["required_passed"])
                self.assertEqual(result["required_failed"], ["macOS Smoke"])

    def test_github_successful_terminal_states(self):
        for conclusion in ("success", "neutral", "skipped"):
            with self.subTest(conclusion=conclusion):
                self.assertTrue(self.report([check("macOS Smoke", conclusion=conclusion)])["required_passed"])

    def test_optional_failure_remains_visible(self):
        result = self.report([check("macOS Smoke"), check("CI Test Telemetry", ident=2,
                                                        conclusion="failure")])
        self.assertTrue(result["required_passed"])
        self.assertEqual(result["other_failed"], ["CI Test Telemetry"])

    def test_legacy_commit_status_is_supported(self):
        requirement = [{"context": "external", "app_id": None}]
        status = {"id": 5, "context": "external", "state": "success"}
        self.assertTrue(self.report([], requirement, [status])["required_passed"])
        status["state"] = "pending"
        self.assertFalse(self.report([], requirement, [status])["required_passed"])

    def test_same_name_legacy_status_cannot_hide_failed_check(self):
        result = self.report([check("macOS Smoke", conclusion="failure")],
                             [{"context": "macOS Smoke", "app_id": None}],
                             [{"id": 9, "context": "macOS Smoke", "state": "success"}])
        self.assertFalse(result["required_passed"])

    def test_ruleset_requirements_join_classic_branch_protection(self):
        classic = {"checks": [{"context": "Linux", "app_id": 15368}]}
        rules = [{"type": "required_status_checks", "parameters": {
            "required_status_checks": [{"context": "macOS", "integration_id": 15368}]}}]
        self.assertEqual(ci_status.requirements(classic, rules), [
            {"context": "Linux", "app_id": 15368},
            {"context": "macOS", "app_id": 15368},
        ])

    def test_required_checks_without_app_constraints_are_retained(self):
        self.assertEqual(ci_status.requirements({"contexts": ["Linux"]}, []),
                         [{"context": "Linux", "app_id": None}])

    def test_head_base_or_state_change_aborts_watch(self):
        original = {"headRefOid": "a" * 40, "baseRefName": "main", "state": "OPEN"}
        ci_status.check_identity(original, original)
        for key, value in (("headRefOid", "b" * 40), ("baseRefName", "other"), ("state", "MERGED")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci_status.check_identity(original, {**original, key: value})


class CommandTests(unittest.TestCase):
    identity = {"headRefOid": "a" * 40, "baseRefName": "main", "state": "OPEN"}

    def replies(self, *, optional=None):
        checks = [check("macOS Smoke")]
        if optional:
            checks.append(optional)
        return [self.identity, {"checks": [{"context": "macOS Smoke", "app_id": 15368}]},
                [[], []], [{"check_runs": checks[:1]}, {"check_runs": checks[1:]}],
                [[], []], self.identity]

    def run_cli(self, replies, *args):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(ci_status, "gh", side_effect=replies) as gh, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = ci_status.main(["1638", *args])
        return code, output.getvalue(), errors.getvalue(), gh

    def test_cli_paginates_and_exits_with_optional_telemetry_pending(self):
        code, output, _, gh = self.run_cli(self.replies(optional=check(
            "CI Test Telemetry", ident=2, status="queued", conclusion=None)), "--watch")
        self.assertEqual(code, 0)
        self.assertIn('"other_pending": ["CI Test Telemetry"]', output)
        self.assertIn("--paginate", gh.call_args_list[2].args)
        self.assertIn("--paginate", gh.call_args_list[3].args)
        self.assertIn("--paginate", gh.call_args_list[4].args)

    def test_cli_does_not_silently_accept_optional_failure(self):
        self.assertEqual(self.run_cli(self.replies(optional=check(
            "CI Test Telemetry", ident=2, conclusion="failure")))[0], 1)

    def test_api_failure_aborts_without_a_report(self):
        for index in range(1, 6):
            with self.subTest(index=index):
                replies = self.replies()
                replies[index] = RuntimeError("HTTP 403")
                code, output, errors, _ = self.run_cli(replies)
                self.assertEqual(code, 3)
                self.assertEqual(output, "")
                self.assertIn("HTTP 403", errors)

    def test_moving_head_during_reads_aborts_without_a_report(self):
        replies = self.replies()
        replies[-1] = {**self.identity, "headRefOid": "b" * 40}
        code, output, _, _ = self.run_cli(replies)
        self.assertEqual(code, 3)
        self.assertEqual(output, "")

    def test_expected_head_mismatch_aborts_before_polling(self):
        code, output, _, gh = self.run_cli([self.identity], "--head", "b" * 40)
        self.assertEqual(code, 3)
        self.assertEqual(output, "")
        self.assertEqual(gh.call_count, 1)

    def test_pending_and_missing_requirements_exit_two(self):
        for check_runs in ([], [check("macOS Smoke", status="queued", conclusion=None)]):
            replies = self.replies()
            replies[3] = [{"check_runs": check_runs}]
            self.assertEqual(self.run_cli(replies)[0], 2)


if __name__ == "__main__":
    unittest.main()
