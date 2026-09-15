"""Exact-base retarget dispatch and receipt controls."""

from __future__ import annotations

from collections import defaultdict
import unittest

from scripts import ci_retarget_validation as retarget


HEAD = "a" * 40
BASE = "b" * 40
TOKEN = "34910356924-1"
REPOSITORY = "Chelis-Lang/chelis"


def pull() -> dict:
    return {
        "state": "open",
        "head": {"sha": HEAD},
        "base": {"sha": BASE, "ref": "main"},
    }


def run_payload(label: str, *, conclusion: str = "success") -> dict:
    return {
        "workflow_runs": [
            {
                "display_title": f"Retarget {label} {TOKEN}",
                "head_sha": BASE,
                "status": "completed",
                "conclusion": conclusion,
                "html_url": f"https://example.invalid/{label.lower()}",
            }
        ]
    }


class FakeApi:
    def __init__(self, *, ci: str = "success", hull: str = "success") -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses = defaultdict(list)
        self.responses[("GET", f"repos/{REPOSITORY}/pulls/2082")].append(pull())
        self.responses[("POST", f"repos/{REPOSITORY}/check-runs")].append(
            {"id": 17}
        )
        self.responses[
            (
                "GET",
                f"repos/{REPOSITORY}/actions/workflows/ci.yml/runs"
                "?event=workflow_dispatch&branch=main&per_page=20",
            )
        ].append(run_payload("CI", conclusion=ci))
        self.responses[
            (
                "GET",
                f"repos/{REPOSITORY}/actions/workflows/conformance.yml/runs"
                "?event=workflow_dispatch&branch=main&per_page=20",
            )
        ].append(run_payload("Hull", conclusion=hull))

    def __call__(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict | None:
        self.calls.append((method, path, payload))
        queued = self.responses[(method, path)]
        if queued:
            return queued.pop(0)
        return None


class RetargetValidationTests(unittest.TestCase):
    def test_exact_retarget_dispatches_both_workflows_and_closes_receipt(self) -> None:
        api = FakeApi()
        result = retarget.run(
            repository=REPOSITORY,
            pr_number=2082,
            expected_head_sha=HEAD,
            expected_base_sha=BASE,
            expected_base_ref="main",
            token=TOKEN,
            api=api,
            sleeper=lambda _: None,
            max_polls=1,
        )
        self.assertEqual(
            result,
            {
                "CI": "https://example.invalid/ci",
                "Hull": "https://example.invalid/hull",
            },
        )
        dispatches = [
            call for call in api.calls if call[1].endswith("/dispatches")
        ]
        self.assertEqual(len(dispatches), 2)
        create_index = next(
            index
            for index, call in enumerate(api.calls)
            if call[0] == "POST" and call[1].endswith("/check-runs")
        )
        self.assertTrue(
            all(api.calls.index(dispatch) > create_index for dispatch in dispatches)
        )
        for _, path, body in dispatches:
            self.assertIn(
                path.rsplit("/", 2)[-2],
                {"ci.yml", "conformance.yml"},
            )
            self.assertEqual(body["ref"], "main")
            self.assertEqual(
                body["inputs"],
                {
                    "pr_number": "2082",
                    "expected_head_sha": HEAD,
                    "expected_base_sha": BASE,
                    "retarget_token": TOKEN,
                },
            )
        update = next(
            call for call in api.calls
            if call[0] == "PATCH" and call[1].endswith("/check-runs/17")
        )
        self.assertEqual(update[2]["status"], "completed")
        self.assertEqual(update[2]["conclusion"], "success")

    def test_failed_or_missing_workflow_fails_the_head_receipt(self) -> None:
        for api, message in (
            (FakeApi(hull="failure"), "Hull"),
            (FakeApi(), "did not appear"),
        ):
            if message == "did not appear":
                api.responses[
                    (
                        "GET",
                        f"repos/{REPOSITORY}/actions/workflows/ci.yml/runs"
                        "?event=workflow_dispatch&branch=main&per_page=20",
                    )
                ] = [{"workflow_runs": []}]
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                retarget.run(
                    repository=REPOSITORY,
                    pr_number=2082,
                    expected_head_sha=HEAD,
                    expected_base_sha=BASE,
                    expected_base_ref="main",
                    token=TOKEN,
                    api=api,
                    sleeper=lambda _: None,
                    max_polls=1,
                )
            update = next(
                call for call in api.calls
                if call[0] == "PATCH" and call[1].endswith("/check-runs/17")
            )
            self.assertEqual(update[2]["conclusion"], "failure")

    def test_stale_or_malformed_event_cannot_create_a_receipt_or_dispatch(self) -> None:
        for field, value, message in (
            ("expected_head_sha", "d" * 40, "stale head"),
            ("expected_base_sha", "d" * 40, "stale base"),
            ("expected_base_ref", "release", "stale base ref"),
            ("token", "../bad", "token"),
        ):
            api = FakeApi()
            kwargs = {
                "repository": REPOSITORY,
                "pr_number": 2082,
                "expected_head_sha": HEAD,
                "expected_base_sha": BASE,
                "expected_base_ref": "main",
                "token": TOKEN,
                "api": api,
                "sleeper": lambda _: None,
                "max_polls": 1,
            }
            kwargs[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, message
            ):
                retarget.run(**kwargs)
            self.assertFalse(
                any(
                    path.endswith("/check-runs") or path.endswith("/dispatches")
                    for _, path, _ in api.calls
                )
            )


if __name__ == "__main__":
    unittest.main()
