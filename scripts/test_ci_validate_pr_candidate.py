"""Exact-head and exact-base validation for manual PR package expansion."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import ci_change_owned as owned
from scripts import ci_validate_pr_candidate as candidate


HEAD = "a" * 40
BASE = "b" * 40
MERGE = "c" * 40


def payload(*, state: str = "open", head: str = HEAD, base: str = BASE) -> dict:
    return {
        "state": state,
        "head": {"sha": head},
        "base": {"sha": base},
    }


def plan() -> dict:
    result = {
        "version": 1,
        "mode": "pull_request",
        "base_sha": BASE,
        "candidate_sha": MERGE,
        "event_pr_head": HEAD,
        "changed_records": [],
        "path_dispositions": [],
        "target_dispositions": [],
        "selected_packages": [],
        "eligible_targets": [],
        "change_owned": [],
        "package_expansion": [],
        "standing_targets": [],
        "manual_only_targets": [],
        "target_exclusions": [],
        "test_exclusions": [],
        "shards": {
            "change_owned": owned.shard_map([]),
            "package_expansion": owned.shard_map([]),
        },
    }
    owned.attach_plan_digest(result)
    return result


class CandidateValidationTests(unittest.TestCase):
    def test_initial_and_final_candidate_validation_pass(self) -> None:
        self.assertEqual(
            candidate.validate_candidate(payload(), expected_head_sha=HEAD),
            (HEAD, BASE),
        )
        self.assertEqual(
            candidate.validate_candidate(
                payload(),
                expected_head_sha=HEAD,
                expected_base_sha=BASE,
            ),
            (HEAD, BASE),
        )
        self.assertEqual(
            candidate.validate_candidate(
                payload(),
                expected_head_sha=HEAD,
                plan=plan(),
            ),
            (HEAD, BASE),
        )

    def test_malformed_closed_stale_and_missing_pr_data_fail(self) -> None:
        cases = [
            ("malformed", payload(), "A" * 40, "lowercase 40-character"),
            ("closed", payload(state="closed"), HEAD, "is not open"),
            ("stale head", payload(head="d" * 40), HEAD, "stale head"),
            ("missing state", {"head": {"sha": HEAD}, "base": {"sha": BASE}}, HEAD, "state"),
            ("missing head", {"state": "open", "base": {"sha": BASE}}, HEAD, "head.sha"),
            ("missing base", {"state": "open", "head": {"sha": HEAD}}, HEAD, "base.sha"),
        ]
        for label, pr, expected, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.validate_candidate(pr, expected_head_sha=expected)

    def test_expected_base_is_exact_and_fail_closed(self) -> None:
        for expected, message in (
            ("B" * 40, "expected_base_sha"),
            ("d" * 40, "stale base"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.validate_candidate(
                    payload(),
                    expected_head_sha=HEAD,
                    expected_base_sha=expected,
                )

    def test_final_validation_rejects_stale_base_head_and_tampered_plan(self) -> None:
        stale_base = copy.deepcopy(plan())
        stale_head = copy.deepcopy(plan())
        stale_head["event_pr_head"] = "d" * 40
        owned.attach_plan_digest(stale_head)
        tampered = copy.deepcopy(plan())
        tampered["base_sha"] = "d" * 40
        for label, pr, planned, message in (
            ("base", payload(base="d" * 40), stale_base, "stale base"),
            ("head", payload(), stale_head, "plan head"),
            ("digest", payload(), tampered, "plan digest mismatch"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.validate_candidate(
                    pr,
                    expected_head_sha=HEAD,
                    plan=planned,
                )

    def test_cli_fetches_one_pr_payload_and_validates_optional_plan(self) -> None:
        runner = mock.Mock(
            return_value=subprocess.CompletedProcess(
                ["gh"], 0, json.dumps(payload()), ""
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_bytes(owned.canonical_json(plan()))
            result = candidate.run(
                repository="Chelis-Lang/chelis",
                pr_number=2071,
                expected_head_sha=HEAD,
                plan_path=plan_path,
                runner=runner,
            )
        self.assertEqual(result, (HEAD, BASE))
        runner.assert_called_once_with(
            [
                "gh",
                "api",
                "repos/Chelis-Lang/chelis/pulls/2071",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_checkout_validation_requires_exact_two_parent_candidate(self) -> None:
        runner = mock.Mock(
            return_value=subprocess.CompletedProcess(
                ["git"],
                0,
                f"{MERGE} {BASE} {HEAD}\n",
                "",
            )
        )
        self.assertEqual(
            candidate.validate_checkout(
                expected_head_sha=HEAD,
                expected_base_sha=BASE,
                runner=runner,
            ),
            MERGE,
        )
        runner.assert_called_once_with(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for row, message in (
            (f"{MERGE} {BASE}\n", "exactly two parents"),
            (f"{MERGE} {HEAD} {BASE}\n", "first parent"),
            (f"{MERGE} {BASE} {'d' * 40}\n", "second parent"),
        ):
            with self.subTest(row=row), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.validate_checkout(
                    expected_head_sha=HEAD,
                    expected_base_sha=BASE,
                    runner=mock.Mock(
                        return_value=subprocess.CompletedProcess(
                            ["git"], 0, row, ""
                        )
                    ),
                )

    def test_cli_rejects_invalid_pr_number_and_gh_or_json_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            candidate.run(
                repository="Chelis-Lang/chelis",
                pr_number=0,
                expected_head_sha=HEAD,
            )
        for error, message in (
            (
                subprocess.CalledProcessError(
                    1, ["gh"], output="", stderr="not found"
                ),
                "cannot read pull request",
            ),
            (
                subprocess.CompletedProcess(["gh"], 0, "{", ""),
                "invalid pull request JSON",
            ),
        ):
            runner = mock.Mock(side_effect=error) if isinstance(
                error, BaseException
            ) else mock.Mock(return_value=error)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.run(
                    repository="Chelis-Lang/chelis",
                    pr_number=2071,
                    expected_head_sha=HEAD,
                    runner=runner,
                )


if __name__ == "__main__":
    unittest.main()
