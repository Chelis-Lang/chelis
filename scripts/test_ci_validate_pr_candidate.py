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
BASE_REF = "main"


def payload(
    *,
    state: str = "open",
    head: str = HEAD,
    base: str = BASE,
    base_ref: str = BASE_REF,
) -> dict:
    return {
        "state": state,
        "head": {"sha": head},
        "base": {"sha": base, "ref": base_ref},
    }


def plan() -> dict:
    baseline = owned.load_duration_baseline()
    change_owned_shards, change_owned_planning = owned.change_owned_shard_plan(
        [],
        baseline,
    )
    expansion_shards, expansion_planning = (
        owned.package_expansion_shard_plan([], baseline)
    )
    result = {
        "version": owned.PLAN_VERSION,
        "mode": "pull_request",
        "base_sha": BASE,
        "candidate_sha": MERGE,
        "event_pr_head": HEAD,
        "config_digest": "d" * 64,
        "changed_records": [],
        "path_dispositions": [],
        "target_dispositions": [
            owned.package_expansion_execution_disposition(
                expansion_shards,
                expansion_planning,
            )
        ],
        "selected_packages": [],
        "eligible_targets": [],
        "target_features": {},
        "change_owned": [],
        "package_expansion": [],
        "standing_targets": [],
        "standing_coverage_reuse": [],
        "manual_only_targets": [],
        "manual_gate_targets": [],
        "target_exclusions": [],
        "test_exclusions": [],
        "shard_planning": {
            "change_owned": change_owned_planning,
            "package_expansion": {
                "algorithm": owned.PACKAGE_EXPANSION_COMPATIBILITY_ALGORITHM,
            },
        },
        "shards": {
            "change_owned": change_owned_shards,
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
                expected_base_ref=BASE_REF,
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

    def test_expected_base_ref_detects_retarget_without_binding_tip(self) -> None:
        self.assertEqual(
            candidate.validate_candidate(
                payload(base="d" * 40),
                expected_head_sha=HEAD,
                expected_base_ref=BASE_REF,
            ),
            (HEAD, "d" * 40),
        )
        for expected, message in (
            ("", "expected_base_ref"),
            ("release", "base retarget"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, message
            ):
                candidate.validate_candidate(
                    payload(),
                    expected_head_sha=HEAD,
                    expected_base_ref=expected,
                )

    def test_final_validation_allows_same_target_advance_but_rejects_head_and_tampering(
        self,
    ) -> None:
        stale_head = copy.deepcopy(plan())
        stale_head["event_pr_head"] = "d" * 40
        owned.attach_plan_digest(stale_head)
        tampered = copy.deepcopy(plan())
        tampered["base_sha"] = "d" * 40
        for label, pr, planned, message in (
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
        self.assertEqual(
            candidate.validate_candidate(
                payload(base="d" * 40),
                expected_head_sha=HEAD,
                expected_base_ref=BASE_REF,
                plan=plan(),
            ),
            (HEAD, "d" * 40),
        )

    def test_cli_fetches_one_pr_payload_and_validates_optional_plan(self) -> None:
        runner = mock.Mock(
            return_value=subprocess.CompletedProcess(
                ["gh"], 0, json.dumps(payload()), ""
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            github_output = Path(tmp) / "github-output"
            plan_path.write_bytes(owned.canonical_json(plan()))
            result = candidate.run(
                repository="Chelis-Lang/chelis",
                pr_number=2071,
                expected_head_sha=HEAD,
                expected_base_ref=BASE_REF,
                github_output=github_output,
                plan_path=plan_path,
                runner=runner,
            )
            output_lines = github_output.read_text(
                encoding="utf-8"
            ).splitlines()
        self.assertEqual(result, (HEAD, BASE))
        self.assertEqual(
            output_lines,
            [
                f"pr_head_sha={HEAD}",
                f"base_sha={BASE}",
                f"base_ref={BASE_REF}",
            ],
        )
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

    def test_cli_separates_live_base_ref_from_checked_out_base_parent(self) -> None:
        runner = mock.Mock(
            side_effect=[
                subprocess.CompletedProcess(
                    ["gh"], 0, json.dumps(payload(base="d" * 40)), ""
                ),
                subprocess.CompletedProcess(
                    ["git"], 0, f"{MERGE} {BASE} {HEAD}\n", ""
                ),
            ]
        )
        self.assertEqual(
            candidate.run(
                repository="Chelis-Lang/chelis",
                pr_number=2071,
                expected_head_sha=HEAD,
                expected_base_ref=BASE_REF,
                checkout_base_sha=BASE,
                validate_checkout_parents=True,
                runner=runner,
            ),
            (HEAD, "d" * 40),
        )

    def test_checkout_must_equal_the_candidate_recorded_by_the_plan(self) -> None:
        planned = plan()
        planned["candidate_sha"] = "d" * 40
        owned.attach_plan_digest(planned)
        runner = mock.Mock(
            side_effect=[
                subprocess.CompletedProcess(
                    ["gh"], 0, json.dumps(payload()), ""
                ),
                subprocess.CompletedProcess(
                    ["git"], 0, f"{MERGE} {BASE} {HEAD}\n", ""
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_bytes(owned.canonical_json(planned))
            with self.assertRaisesRegex(ValueError, "plan candidate"):
                candidate.run(
                    repository="Chelis-Lang/chelis",
                    pr_number=2071,
                    expected_head_sha=HEAD,
                    expected_base_ref=BASE_REF,
                    checkout_base_sha=BASE,
                    validate_checkout_parents=True,
                    plan_path=plan_path,
                    runner=runner,
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
                    expected_base_ref=BASE_REF,
                    runner=runner,
                )


if __name__ == "__main__":
    unittest.main()
