"""Contract tests for the OpenSpec merge worker.

The worker holds the only write token in the system. Its job is to refuse,
and to merge exactly one commit when every refusal reason is absent.

Three real properties of this repository's API responses drive the design,
and each has a test here:

1. Check runs attach to the pull request HEAD sha, not to the test-merge
   sha. Binding the merge to the head sha is therefore the same commit the
   checks ran on.
2. A required context appears MANY times on one commit -- `Changelog` was
   observed with three `success` runs and one `cancelled` run. Neither
   `all()` nor `any()` is correct; only the latest run per context counts.
3. The legacy combined-status endpoint returns zero statuses and reports
   `state: "pending"` for commits whose checks all passed. Reading it as
   the answer would either never merge or merge on a misreading.

Four more properties were measured against the live API on 2026-09-08 and
each has its own section below:

4. `GET .../branches/{branch}/protection` requires the `Administration`
   permission, which a workflow `permissions:` block cannot grant, because
   `administration` is not one of the keys GITHUB_TOKEN accepts. The
   required-check set must come from `GET .../branches/{branch}`, which
   needs only `Contents: read` and carries the same `checks` array.
5. A commit status carries no app, so it can never satisfy a context that
   protection pins to an app. Anyone with write access can post one.
6. A verdict authorizes a `(base ref, base sha, head sha)` triple, not a
   pull request number. Retargeting the base after the verdict changes
   what a squash merge would land.
7. A check-run name is not provenance. The strict validation requirement
   is pinned to a successful `pull_request` run of one workflow FILE on
   the exact head sha.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "openspec_merge.py"

SPEC = importlib.util.spec_from_file_location("openspec_merge", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load the merge module: {MODULE_PATH}")
merge_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_module
SPEC.loader.exec_module(merge_module)

REPO = "Chelis-Lang/chelis"
HEAD = "a" * 40
BASE = "b" * 40
MOVED_BASE = "e" * 40
BASE_REF = "main"
ACTIONS_APP = 15368
REQUIRED = ("Lint and Unit Tests (Linux)", "Integration Tests (Linux)")
VALIDATE_WORKFLOW = "openspec-autoland-validate.yml"
VALIDATION_CONTEXT = "OpenSpec Strict Validation"


def check_run(
    name: str,
    conclusion: str | None = "success",
    *,
    identifier: int = 1,
    status: str = "completed",
    app: int = ACTIONS_APP,
    started: str = "2026-09-08T21:00:00Z",
) -> dict:
    return {
        "id": identifier,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started,
        "app": {"id": app},
    }


def workflow_run(
    *,
    identifier: int = 100,
    conclusion: str | None = "success",
    status: str = "completed",
    head_sha: str = HEAD,
    event: str = "pull_request",
    attempt: int = 1,
) -> dict:
    return {
        "id": identifier,
        "path": f".github/workflows/{VALIDATE_WORKFLOW}",
        "event": event,
        "status": status,
        "conclusion": conclusion,
        "head_sha": head_sha,
        "run_attempt": attempt,
    }


def protection_payload(
    contexts: tuple[str, ...] = REQUIRED, app: int | None = ACTIONS_APP
) -> dict:
    if app is None:
        return {"required_status_checks": {"contexts": list(contexts)}}
    return {
        "required_status_checks": {
            "checks": [{"context": name, "app_id": app} for name in contexts]
        }
    }


class FakeApi:
    """A stand-in for `gh api` that records every call."""

    def __init__(
        self,
        *,
        pull: dict | None = None,
        protection: object = None,
        branch: object = None,
        runs: list[dict] | None = None,
        statuses: dict | None = None,
        merge_result: object = None,
        pull_sequence: list[dict] | None = None,
        runs_sequence: list[list[dict]] | None = None,
        run_pages: list[list[dict]] | None = None,
        status_pages: list[list[dict]] | None = None,
        workflow_runs: object = None,
        compare: object = None,
    ) -> None:
        self.pull = pull or self.default_pull()
        self.pull_sequence = pull_sequence
        self.protection = (
            protection if protection is not None else protection_payload()
        )
        # By default the branch endpoint reports the same protection the
        # dedicated endpoint does, which is what the live API returns.
        if branch is not None:
            self.branch: object = branch
        else:
            # An unreadable `protection` models the endpoint being out of
            # reach, which is the live case for GITHUB_TOKEN. The branch
            # endpoint is a different permission, so it still answers.
            source = (
                self.protection
                if isinstance(self.protection, dict)
                else protection_payload()
            )
            self.branch = {
                "name": BASE_REF,
                "protected": True,
                "protection": {
                    "enabled": True,
                    **{
                        key: value
                        for key, value in source.items()
                        if key == "required_status_checks"
                    },
                },
            }
        self.runs = runs if runs is not None else [check_run(n) for n in REQUIRED]
        self.runs_sequence = runs_sequence
        self.run_pages = run_pages
        self.statuses = statuses or {"state": "pending", "statuses": []}
        self.status_pages = status_pages
        self.workflow_runs = (
            workflow_runs if workflow_runs is not None else [workflow_run()]
        )
        self.compare = compare if compare is not None else {"status": "ahead"}
        self.merge_result = merge_result
        self.calls: list[list[str]] = []
        self.pull_reads = 0
        self.runs_reads = 0

    @staticmethod
    def default_pull(**overrides: object) -> dict:
        pull = {
            "number": 7,
            "state": "open",
            "draft": False,
            "merged": False,
            "mergeable": True,
            "mergeable_state": "clean",
            "head": {"sha": HEAD, "repo": {"full_name": REPO}},
            "base": {"sha": BASE, "ref": BASE_REF},
        }
        pull.update(overrides)  # type: ignore[arg-type]
        return pull

    @staticmethod
    def _page(target: str) -> int:
        marker = "&page="
        if marker in target:
            return int(target.split(marker, 1)[1].split("&", 1)[0])
        return 1

    def _resolve(self, value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(list(arguments))
        target = arguments[-1] if arguments else ""
        if "--method" in arguments and "PUT" in arguments:
            if isinstance(self.merge_result, Exception):
                raise self.merge_result
            return json.dumps(self.merge_result or {"merged": True, "sha": "z" * 40})
        if "/compare/" in target:
            return json.dumps(self._resolve(self.compare))
        if "/actions/workflows/" in target:
            runs = self._resolve(self.workflow_runs)
            assert isinstance(runs, list)
            return json.dumps({"total_count": len(runs), "workflow_runs": runs})
        if "/pulls/" in target and "/merge" not in target:
            self.pull_reads += 1
            if self.pull_sequence:
                index = min(self.pull_reads - 1, len(self.pull_sequence) - 1)
                return json.dumps(self.pull_sequence[index])
            return json.dumps(self.pull)
        if target.endswith("/protection"):
            return json.dumps(self._resolve(self.protection))
        if "/branches/" in target:
            return json.dumps(self._resolve(self.branch))
        if "/check-runs" in target:
            self.runs_reads += 1
            page = self._page(target)
            if self.run_pages is not None:
                runs = (
                    self.run_pages[page - 1] if page <= len(self.run_pages) else []
                )
            else:
                if self.runs_sequence:
                    index = min(self.runs_reads - 1, len(self.runs_sequence) - 1)
                    runs = self.runs_sequence[index]
                else:
                    runs = self.runs
                if page > 1:
                    runs = []
            return json.dumps({"total_count": len(runs), "check_runs": runs})
        if "/status" in target:
            page = self._page(target)
            if self.status_pages is not None:
                entries = (
                    self.status_pages[page - 1]
                    if page <= len(self.status_pages)
                    else []
                )
                return json.dumps({"state": "pending", "statuses": entries})
            if page > 1:
                return json.dumps({"state": "pending", "statuses": []})
            return json.dumps(self.statuses)
        raise AssertionError(f"unexpected API call: {arguments}")

    def merged(self) -> bool:
        return any("--method" in call and "PUT" in call for call in self.calls)

    def targets(self) -> str:
        return " ".join(" ".join(call) for call in self.calls)


class RequiredCheckDiscoveryTests(unittest.TestCase):
    def test_required_checks_come_from_branch_protection(self) -> None:
        api = FakeApi()
        checks = merge_module.read_required_checks(REPO, BASE_REF, api)
        self.assertEqual([c.context for c in checks], list(REQUIRED))
        self.assertTrue(all(c.app_id == ACTIONS_APP for c in checks))

    def test_the_legacy_contexts_list_is_accepted_when_checks_is_absent(self) -> None:
        api = FakeApi(protection=protection_payload(app=None))
        checks = merge_module.read_required_checks(REPO, BASE_REF, api)
        self.assertEqual([c.context for c in checks], list(REQUIRED))
        self.assertTrue(all(c.app_id is None for c in checks))

    def test_an_empty_required_set_is_never_a_pass(self) -> None:
        """No required checks means nothing was proven, not 'all green'."""
        for protection in (
            {"required_status_checks": {"checks": []}},
            {"required_status_checks": {"contexts": []}},
            {},
        ):
            with self.subTest(protection=protection):
                api = FakeApi(protection=protection)
                with self.assertRaises(merge_module.MergeError):
                    merge_module.read_required_checks(REPO, BASE_REF, api)

    def test_unreadable_protection_is_operational_not_a_pass(self) -> None:
        api = FakeApi(
            protection=merge_module.MergeError("404"),
            branch=merge_module.MergeError("404"),
        )
        with self.assertRaises(merge_module.MergeError):
            merge_module.read_required_checks(REPO, BASE_REF, api)

    # --- Measured fact 4: the protection endpoint is out of reach ---

    def test_the_branch_endpoint_alone_discovers_the_required_set(self) -> None:
        """`GET .../branches/{b}` needs Contents: read, which the token has.

        `GET .../branches/{b}/protection` needs `Administration: read`, and
        a workflow `permissions:` block has no `administration` key, so the
        worker must never depend on it.
        """
        api = FakeApi(
            protection=merge_module.MergeError(
                "403 Resource not accessible by integration"
            )
        )
        checks = merge_module.read_required_checks(REPO, BASE_REF, api)
        self.assertEqual([c.context for c in checks], list(REQUIRED))
        self.assertTrue(all(c.app_id == ACTIONS_APP for c in checks))

    def test_the_branch_endpoint_is_tried_before_the_protection_endpoint(self) -> None:
        api = FakeApi()
        merge_module.read_required_checks(REPO, BASE_REF, api)
        self.assertNotIn("/protection", api.targets())

    def test_an_unprotected_base_branch_is_refused_by_name(self) -> None:
        api = FakeApi(branch={"name": BASE_REF, "protected": False})
        with self.assertRaises(merge_module.MergeError) as caught:
            merge_module.read_required_checks(REPO, BASE_REF, api)
        self.assertIn("not protected", str(caught.exception))

    def test_both_discovery_paths_failing_names_the_activation_blocker(self) -> None:
        api = FakeApi(
            branch=merge_module.MergeError("403 Resource not accessible"),
            protection=merge_module.MergeError("403 Resource not accessible"),
        )
        with self.assertRaises(merge_module.MergeError) as caught:
            merge_module.read_required_checks(REPO, BASE_REF, api)
        message = str(caught.exception)
        self.assertIn("Administration", message)
        self.assertIn("Contents", message)

    def test_a_base_ref_that_is_not_a_branch_name_never_reaches_the_api(self) -> None:
        api = FakeApi()
        for value in ("", "../../etc", "main?x=1", "main#f", "main\n", "-main"):
            with self.subTest(value=value):
                with self.assertRaises(merge_module.MergeError):
                    merge_module.read_required_checks(REPO, value, api)
        self.assertEqual(api.calls, [])

    def test_an_ordinary_nested_branch_name_is_accepted(self) -> None:
        api = FakeApi(
            branch={
                "name": "release/1.x",
                "protected": True,
                "protection": protection_payload(),
            }
        )
        checks = merge_module.read_required_checks(REPO, "release/1.x", api)
        self.assertEqual(len(checks), len(REQUIRED))


class CheckEvaluationTests(unittest.TestCase):
    def required(self, app: int | None = ACTIONS_APP) -> tuple:
        return tuple(
            merge_module.RequiredCheck(context=name, app_id=app) for name in REQUIRED
        )

    def evaluate(
        self,
        runs: list[dict],
        statuses: dict | None = None,
        app: int | None = ACTIONS_APP,
    ):
        api = FakeApi(runs=runs, statuses=statuses)
        observed = merge_module.read_check_state(REPO, HEAD, api)
        return merge_module.failing_checks(self.required(app), observed)

    def test_all_required_contexts_successful_passes(self) -> None:
        self.assertEqual(self.evaluate([check_run(n) for n in REQUIRED]), ())

    def test_the_latest_run_per_context_decides(self) -> None:
        """Observed live: three success runs and one cancelled on one sha."""
        runs = [
            check_run(
                REQUIRED[0], "cancelled", identifier=1, started="2026-09-08T21:00:00Z"
            ),
            check_run(
                REQUIRED[0], "success", identifier=2, started="2026-09-08T21:30:00Z"
            ),
            check_run(REQUIRED[1], "success", identifier=3),
        ]
        self.assertEqual(self.evaluate(runs), ())

    def test_a_later_failure_overrides_an_earlier_success(self) -> None:
        runs = [
            check_run(
                REQUIRED[0], "success", identifier=1, started="2026-09-08T21:00:00Z"
            ),
            check_run(
                REQUIRED[0], "failure", identifier=9, started="2026-09-08T22:00:00Z"
            ),
            check_run(REQUIRED[1], "success", identifier=3),
        ]
        failures = self.evaluate(runs)
        self.assertEqual(len(failures), 1)
        self.assertIn(REQUIRED[0], failures[0])

    def test_a_missing_required_context_fails_closed(self) -> None:
        failures = self.evaluate([check_run(REQUIRED[0])])
        self.assertEqual(len(failures), 1)
        self.assertIn("no result", failures[0])

    def test_no_checks_at_all_fails_closed(self) -> None:
        failures = self.evaluate([])
        self.assertEqual(len(failures), len(REQUIRED))

    def test_a_skipped_required_check_satisfies_it(self) -> None:
        """Branch protection treats `skipped` as satisfying a required check.

        This is not a relaxation, it is the measured behaviour of the exact
        class of pull request this feature exists for. On real docs-only
        pull request #1634 -- which merged -- six of the nine required
        contexts reported `skipped`, because `ci.yml` skips its heavy jobs
        for a docs-only diff (chelis#419). Refusing `skipped` would refuse
        every eligible pull request while protection was happy to merge it.
        """
        runs = [
            check_run(REQUIRED[0], "skipped"),
            check_run(REQUIRED[1], "success"),
        ]
        self.assertEqual(self.evaluate(runs), ())

    def test_a_skipped_result_never_satisfies_an_explicitly_required_context(
        self,
    ) -> None:
        """`skipped` is accepted for protection's own contexts, not for ours.

        `--require-context` names the strict OpenSpec validation, which is
        the one check that must actually have run: a skipped validation
        proves nothing about the documents being merged. Protection's
        contexts are different -- they are skipped by design on a docs-only
        diff, and protection accepts that.
        """
        required = (
            merge_module.RequiredCheck(
                context="OpenSpec Strict Validation",
                app_id=ACTIONS_APP,
                must_run=True,
            ),
        )
        for conclusion, expected in (("success", 0), ("skipped", 1)):
            with self.subTest(conclusion=conclusion):
                api = FakeApi(
                    runs=[check_run("OpenSpec Strict Validation", conclusion)]
                )
                observed = merge_module.read_check_state(REPO, HEAD, api)
                self.assertEqual(
                    len(merge_module.failing_checks(required, observed)), expected
                )

    def test_a_skipped_check_still_loses_to_a_later_failure(self) -> None:
        runs = [
            check_run(REQUIRED[0], "skipped", identifier=1, started="2026-09-08T21:00:00Z"),
            check_run(REQUIRED[0], "failure", identifier=9, started="2026-09-08T22:00:00Z"),
            check_run(REQUIRED[1], "success"),
        ]
        self.assertEqual(len(self.evaluate(runs)), 1)

    def test_every_non_success_conclusion_fails_closed(self) -> None:
        for conclusion in (
            "failure",
            "cancelled",
            "neutral",
            "timed_out",
            "action_required",
            "stale",
            None,
        ):
            with self.subTest(conclusion=conclusion):
                runs = [
                    check_run(REQUIRED[0], conclusion),
                    check_run(REQUIRED[1], "success"),
                ]
                failures = self.evaluate(runs)
                self.assertEqual(len(failures), 1, conclusion)
                self.assertIn(REQUIRED[0], failures[0])

    def test_an_incomplete_run_fails_closed(self) -> None:
        for status in ("queued", "in_progress", "waiting", "pending"):
            with self.subTest(status=status):
                runs = [
                    check_run(REQUIRED[0], None, status=status),
                    check_run(REQUIRED[1], "success"),
                ]
                self.assertEqual(len(self.evaluate(runs)), 1)

    def test_a_context_from_the_wrong_app_does_not_satisfy_it(self) -> None:
        runs = [
            check_run(REQUIRED[0], "success", app=99999),
            check_run(REQUIRED[1], "success"),
        ]
        failures = self.evaluate(runs)
        self.assertEqual(len(failures), 1)
        self.assertIn(REQUIRED[0], failures[0])

    def test_a_commit_status_can_satisfy_a_legacy_context(self) -> None:
        """Protection that names bare contexts accepts any reporter, so we do."""
        statuses = {
            "state": "success",
            "statuses": [
                {
                    "context": REQUIRED[1],
                    "state": "success",
                    "created_at": "2026-09-08T21:00:00Z",
                }
            ],
        }
        self.assertEqual(
            self.evaluate([check_run(REQUIRED[0])], statuses, app=None), ()
        )

    def test_the_latest_commit_status_per_context_decides(self) -> None:
        statuses = {
            "state": "success",
            "statuses": [
                {
                    "context": REQUIRED[1],
                    "state": "success",
                    "created_at": "2026-09-08T21:00:00Z",
                },
                {
                    "context": REQUIRED[1],
                    "state": "failure",
                    "created_at": "2026-09-08T22:00:00Z",
                },
            ],
        }
        self.assertEqual(
            len(self.evaluate([check_run(REQUIRED[0])], statuses, app=None)), 1
        )

    def test_an_empty_combined_status_is_not_read_as_the_answer(self) -> None:
        """Live: `state` is "pending" with zero statuses on a green commit."""
        statuses = {"state": "pending", "statuses": []}
        self.assertEqual(self.evaluate([check_run(n) for n in REQUIRED], statuses), ())

    # --- Measured fact 5: a commit status has no app ---

    def test_a_commit_status_cannot_satisfy_an_app_pinned_context(self) -> None:
        """Anyone with write access can post a status; none carries an app.

        Protection pins all nine contexts on `main` to app 15368, so a
        status can never satisfy one. Accepting it would let a contributor
        forge a required context before Actions reports the real one.
        """
        statuses = {
            "state": "success",
            "statuses": [
                {
                    "context": REQUIRED[1],
                    "state": "success",
                    "created_at": "2026-09-08T21:00:00Z",
                }
            ],
        }
        failures = self.evaluate([check_run(REQUIRED[0])], statuses)
        self.assertEqual(len(failures), 1)
        self.assertIn(REQUIRED[1], failures[0])
        self.assertIn("app", failures[0])

    def test_a_result_with_no_app_never_satisfies_an_app_pinned_context(self) -> None:
        observed = {
            REQUIRED[0]: merge_module.CheckResult(
                context=REQUIRED[0],
                app_id=None,
                status="completed",
                conclusion="success",
                order=(1, "", 1),
            )
        }
        required = (merge_module.RequiredCheck(REQUIRED[0], ACTIONS_APP),)
        self.assertEqual(len(merge_module.failing_checks(required, observed)), 1)


class PaginationTests(unittest.TestCase):
    """A truncated read must never be reported as a complete one."""

    def test_exhausting_the_check_run_pages_is_an_error_not_an_answer(self) -> None:
        full = [check_run(f"context {i}", identifier=i) for i in range(100)]
        api = FakeApi(run_pages=[full] * (merge_module.MAX_PAGES + 2))
        with self.assertRaises(merge_module.MergeError) as caught:
            merge_module.read_check_state(REPO, HEAD, api)
        self.assertIn("page", str(caught.exception).lower())

    def test_a_short_final_page_ends_the_read_cleanly(self) -> None:
        full = [check_run(f"context {i}", identifier=i) for i in range(100)]
        api = FakeApi(run_pages=[full, [check_run(REQUIRED[0])]])
        observed = merge_module.read_check_state(REPO, HEAD, api)
        self.assertIn(REQUIRED[0], observed)

    def test_the_combined_status_read_asks_for_a_full_page(self) -> None:
        api = FakeApi()
        merge_module.read_check_state(REPO, HEAD, api)
        status_calls = [c for c in api.calls if "/status" in c[-1]]
        self.assertTrue(status_calls)
        self.assertTrue(any("per_page=100" in c[-1] for c in status_calls))

    def test_exhausting_the_status_pages_is_an_error_not_an_answer(self) -> None:
        full = [
            {
                "context": f"context {i}",
                "state": "success",
                "created_at": "2026-09-08T21:00:00Z",
            }
            for i in range(100)
        ]
        api = FakeApi(status_pages=[full] * (merge_module.MAX_PAGES + 2))
        with self.assertRaises(merge_module.MergeError):
            merge_module.read_check_state(REPO, HEAD, api)


class WorkflowProvenanceTests(unittest.TestCase):
    """Measured fact 7: a check-run NAME is not provenance.

    Any Actions workflow can publish a job named `OpenSpec Strict
    Validation`, and any collaborator can post a commit status with that
    context. The requirement is pinned to a successful `pull_request` run
    of one workflow FILE on the exact head sha instead.
    """

    def failures(self, api: FakeApi, sha: str = HEAD) -> tuple[str, ...]:
        return merge_module.workflow_failures(REPO, VALIDATE_WORKFLOW, sha, api)

    def test_a_successful_run_on_the_head_satisfies_the_requirement(self) -> None:
        self.assertEqual(self.failures(FakeApi()), ())

    def test_no_run_at_all_refuses(self) -> None:
        failures = self.failures(FakeApi(workflow_runs=[]))
        self.assertEqual(len(failures), 1)
        self.assertIn(VALIDATE_WORKFLOW, failures[0])

    def test_a_failed_run_refuses(self) -> None:
        api = FakeApi(workflow_runs=[workflow_run(conclusion="failure")])
        self.assertEqual(len(self.failures(api)), 1)

    def test_an_incomplete_run_refuses(self) -> None:
        for status in ("queued", "in_progress", "waiting"):
            with self.subTest(status=status):
                api = FakeApi(
                    workflow_runs=[workflow_run(conclusion=None, status=status)]
                )
                self.assertEqual(len(self.failures(api)), 1)

    def test_a_run_on_another_commit_does_not_satisfy_it(self) -> None:
        api = FakeApi(workflow_runs=[workflow_run(head_sha="f" * 40)])
        self.assertEqual(len(self.failures(api)), 1)

    def test_a_run_from_another_event_does_not_satisfy_it(self) -> None:
        api = FakeApi(workflow_runs=[workflow_run(event="push")])
        self.assertEqual(len(self.failures(api)), 1)

    def test_the_latest_run_decides_over_an_earlier_one(self) -> None:
        api = FakeApi(
            workflow_runs=[
                workflow_run(identifier=1, conclusion="success"),
                workflow_run(identifier=2, conclusion="failure"),
            ]
        )
        self.assertEqual(len(self.failures(api)), 1)

    def test_a_later_success_clears_an_earlier_failure(self) -> None:
        api = FakeApi(
            workflow_runs=[
                workflow_run(identifier=1, conclusion="failure"),
                workflow_run(identifier=2, conclusion="success"),
            ]
        )
        self.assertEqual(self.failures(api), ())

    def test_the_latest_attempt_of_one_run_decides(self) -> None:
        api = FakeApi(
            workflow_runs=[
                workflow_run(identifier=5, conclusion="failure", attempt=1),
                workflow_run(identifier=5, conclusion="success", attempt=2),
            ]
        )
        self.assertEqual(self.failures(api), ())

    def test_an_unreadable_response_is_operational_not_a_pass(self) -> None:
        api = FakeApi(workflow_runs=merge_module.MergeError("403"))
        with self.assertRaises(merge_module.MergeError):
            self.failures(api)

    def test_the_workflow_name_is_validated_before_it_reaches_a_url(self) -> None:
        api = FakeApi()
        for value in ("", "../ci.yml", ".github/workflows/x.yml", "x.yml?a=1", "x"):
            with self.subTest(value=value):
                with self.assertRaises(merge_module.MergeError):
                    merge_module.workflow_failures(REPO, value, HEAD, api)
        self.assertEqual(api.calls, [])

    def test_the_request_is_scoped_to_the_file_and_the_head(self) -> None:
        api = FakeApi()
        self.failures(api)
        target = api.calls[-1][-1]
        self.assertIn(f"actions/workflows/{VALIDATE_WORKFLOW}/runs", target)
        self.assertIn(f"head_sha={HEAD}", target)


class EligibilityTests(unittest.TestCase):
    def refusals(self, **overrides: object) -> tuple[str, ...]:
        pull = merge_module.PullRequest.from_payload(FakeApi.default_pull(**overrides))
        return merge_module.eligibility_refusals(pull, REPO, HEAD, BASE_REF)

    def test_an_ordinary_open_pull_request_is_eligible(self) -> None:
        self.assertEqual(self.refusals(), ())

    def test_a_draft_is_refused(self) -> None:
        self.assertTrue(any("draft" in r for r in self.refusals(draft=True)))

    def test_a_closed_or_merged_pull_request_is_refused(self) -> None:
        self.assertTrue(self.refusals(state="closed"))
        self.assertTrue(self.refusals(merged=True))

    def test_a_fork_head_is_refused(self) -> None:
        refusals = self.refusals(
            head={"sha": HEAD, "repo": {"full_name": "someone/fork"}}
        )
        self.assertTrue(any("fork" in r for r in refusals))

    def test_a_missing_head_repository_is_refused(self) -> None:
        self.assertTrue(self.refusals(head={"sha": HEAD, "repo": None}))

    def test_a_moved_head_is_refused(self) -> None:
        refusals = self.refusals(head={"sha": "c" * 40, "repo": {"full_name": REPO}})
        self.assertTrue(any("head" in r for r in refusals))

    def test_an_unmergeable_pull_request_is_refused(self) -> None:
        self.assertTrue(self.refusals(mergeable=False))
        self.assertTrue(self.refusals(mergeable_state="dirty"))

    def test_an_undecided_mergeable_flag_is_refused_for_now(self) -> None:
        """`mergeable` is null while GitHub computes it; that is not yes."""
        self.assertTrue(self.refusals(mergeable=None))

    # --- Measured fact 6: the verdict authorizes a base, not a number ---

    def test_a_retargeted_base_branch_is_refused(self) -> None:
        refusals = self.refusals(base={"sha": BASE, "ref": "release/1.x"})
        self.assertTrue(any("base" in r for r in refusals), refusals)

    def test_the_authorized_base_branch_is_not_a_refusal(self) -> None:
        self.assertEqual(self.refusals(base={"sha": BASE, "ref": BASE_REF}), ())


class BaseMovementTests(unittest.TestCase):
    """The base sha may only move forward on the authorized branch.

    A squash merge lands `merge-base(base, head)...head`. If the base ref
    is rolled back or rewritten, that diff grows to include commits the
    boundary classifier never judged.
    """

    def refusals(self, live_base: str, compare: object = None) -> tuple[str, ...]:
        api = FakeApi(compare=compare)
        pull = merge_module.PullRequest.from_payload(
            FakeApi.default_pull(base={"sha": live_base, "ref": BASE_REF})
        )
        return merge_module.base_movement_refusals(REPO, pull, BASE, api)

    def test_an_unmoved_base_needs_no_comparison(self) -> None:
        api = FakeApi(compare=merge_module.MergeError("must not be called"))
        pull = merge_module.PullRequest.from_payload(FakeApi.default_pull())
        self.assertEqual(merge_module.base_movement_refusals(REPO, pull, BASE, api), ())
        self.assertEqual(api.calls, [])

    def test_a_base_that_moved_forward_is_accepted(self) -> None:
        self.assertEqual(self.refusals(MOVED_BASE, {"status": "ahead"}), ())

    def test_an_identical_base_is_accepted(self) -> None:
        self.assertEqual(self.refusals(MOVED_BASE, {"status": "identical"}), ())

    def test_a_rolled_back_base_is_refused(self) -> None:
        refusals = self.refusals(MOVED_BASE, {"status": "behind"})
        self.assertEqual(len(refusals), 1)
        self.assertIn("behind", refusals[0])

    def test_a_diverged_base_is_refused(self) -> None:
        self.assertEqual(len(self.refusals(MOVED_BASE, {"status": "diverged"})), 1)

    def test_an_unknown_comparison_status_is_refused(self) -> None:
        self.assertEqual(len(self.refusals(MOVED_BASE, {"status": "surprise"})), 1)
        self.assertEqual(len(self.refusals(MOVED_BASE, {})), 1)

    def test_an_unreadable_comparison_is_operational_not_a_pass(self) -> None:
        with self.assertRaises(merge_module.MergeError):
            self.refusals(MOVED_BASE, merge_module.MergeError("404"))

    def test_the_comparison_runs_from_the_authorized_base_to_the_live_one(self) -> None:
        api = FakeApi()
        pull = merge_module.PullRequest.from_payload(
            FakeApi.default_pull(base={"sha": MOVED_BASE, "ref": BASE_REF})
        )
        merge_module.base_movement_refusals(REPO, pull, BASE, api)
        self.assertIn(f"compare/{BASE}...{MOVED_BASE}", api.calls[-1][-1])


class MergeCallTests(unittest.TestCase):
    def test_the_merge_binds_the_exact_head_and_squashes(self) -> None:
        api = FakeApi()
        merge_module.merge_pull_request(REPO, 7, HEAD, api)
        call = api.calls[-1]
        self.assertIn("PUT", call)
        self.assertIn(f"repos/{REPO}/pulls/7/merge", call)
        self.assertIn(f"sha={HEAD}", call)
        self.assertIn("merge_method=squash", call)

    def test_the_merge_never_requests_auto_merge_or_a_bypass(self) -> None:
        api = FakeApi()
        merge_module.merge_pull_request(REPO, 7, HEAD, api)
        flat = api.targets()
        for forbidden in (
            "--auto",
            "--admin",
            "auto_merge",
            "enablePullRequest",
            "review",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, flat)

    def test_a_protection_refusal_is_reported_as_blocked(self) -> None:
        api = FakeApi(merge_result=merge_module.MergeError("405 not mergeable"))
        with self.assertRaises(merge_module.MergeError):
            merge_module.merge_pull_request(REPO, 7, HEAD, api)


class WorkerTests(unittest.TestCase):
    def run_worker(self, api: FakeApi, **kwargs: object) -> tuple[int, str]:
        out = io.StringIO()
        options = {
            "repository": REPO,
            "number": 7,
            "expected_head": HEAD,
            "expected_base_ref": BASE_REF,
            "expected_base_sha": BASE,
            "timeout_seconds": 0,
            "poll_seconds": 0,
            "api": api,
            "sleeper": lambda _seconds: None,
        }
        options.update(kwargs)
        with redirect_stdout(out), redirect_stderr(out):
            status = merge_module.run(**options)  # type: ignore[arg-type]
        return status, out.getvalue()

    def test_a_green_eligible_pull_request_is_merged(self) -> None:
        api = FakeApi()
        status, output = self.run_worker(api)
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())
        self.assertIn("merged", output)

    def test_a_failing_required_check_is_never_merged(self) -> None:
        api = FakeApi(runs=[check_run(REQUIRED[0], "failure"), check_run(REQUIRED[1])])
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn(REQUIRED[0], output)

    def test_a_pending_check_times_out_without_merging(self) -> None:
        api = FakeApi(
            runs=[
                check_run(REQUIRED[0], None, status="in_progress"),
                check_run(REQUIRED[1]),
            ]
        )
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn("re-run", output.lower())

    def test_the_timeout_message_names_only_supported_retries(self) -> None:
        """The autoland workflow declares no `workflow_dispatch` trigger.

        `gh workflow run openspec-autoland.yml` fails outright there, so
        printing it strands a maintainer on a dead command.
        """
        api = FakeApi(
            runs=[
                check_run(REQUIRED[0], None, status="in_progress"),
                check_run(REQUIRED[1]),
            ]
        )
        _, output = self.run_worker(api)
        self.assertNotIn("gh workflow run", output)
        self.assertIn("--allow-empty", output)

    def test_checks_that_turn_green_during_polling_are_merged(self) -> None:
        api = FakeApi(
            runs_sequence=[
                [
                    check_run(REQUIRED[0], None, status="in_progress"),
                    check_run(REQUIRED[1]),
                ],
                [check_run(REQUIRED[0]), check_run(REQUIRED[1])],
            ],
            pull_sequence=[FakeApi.default_pull()] * 6,
        )
        status, output = self.run_worker(api, timeout_seconds=60)
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())

    def test_a_head_that_moves_during_polling_refuses_without_merging(self) -> None:
        moved = FakeApi.default_pull(
            head={"sha": "d" * 40, "repo": {"full_name": REPO}}
        )
        api = FakeApi(pull_sequence=[FakeApi.default_pull(), moved])
        status, output = self.run_worker(api, timeout_seconds=60)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn("head", output)

    def test_a_base_retargeted_during_polling_refuses_without_merging(self) -> None:
        retargeted = FakeApi.default_pull(base={"sha": BASE, "ref": "release/1.x"})
        api = FakeApi(
            pull_sequence=[FakeApi.default_pull(), retargeted],
            runs_sequence=[
                [
                    check_run(REQUIRED[0], None, status="in_progress"),
                    check_run(REQUIRED[1]),
                ],
                [check_run(REQUIRED[0]), check_run(REQUIRED[1])],
            ],
        )
        status, output = self.run_worker(api, timeout_seconds=60)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn("base", output)

    def test_a_base_retargeted_before_the_first_poll_refuses(self) -> None:
        api = FakeApi(pull=FakeApi.default_pull(base={"sha": BASE, "ref": "other"}))
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertNotIn("/branches/", api.targets())

    def test_a_base_rolled_back_before_the_merge_refuses(self) -> None:
        api = FakeApi(
            pull=FakeApi.default_pull(base={"sha": MOVED_BASE, "ref": BASE_REF}),
            compare={"status": "behind"},
        )
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn("behind", output)

    def test_a_base_that_advanced_normally_still_merges(self) -> None:
        api = FakeApi(
            pull=FakeApi.default_pull(base={"sha": MOVED_BASE, "ref": BASE_REF}),
            compare={"status": "ahead"},
        )
        status, output = self.run_worker(api)
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())

    def test_the_pull_request_is_re_read_immediately_before_merging(self) -> None:
        api = FakeApi()
        self.run_worker(api)
        merge_index = next(i for i, c in enumerate(api.calls) if "PUT" in c)
        last_read = max(
            i
            for i, c in enumerate(api.calls)
            if "/pulls/7" in c[-1] and "PUT" not in c
        )
        self.assertLess(last_read, merge_index)
        self.assertGreaterEqual(api.pull_reads, 2)

    def test_a_draft_is_refused_before_any_check_is_read(self) -> None:
        api = FakeApi(pull=FakeApi.default_pull(draft=True))
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertNotIn("check-runs", api.targets())

    def test_a_fork_is_refused_and_nothing_is_written(self) -> None:
        api = FakeApi(
            pull=FakeApi.default_pull(
                head={"sha": HEAD, "repo": {"full_name": "someone/fork"}}
            )
        )
        status, _ = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())

    def test_an_empty_required_set_refuses_rather_than_merging(self) -> None:
        api = FakeApi(protection={"required_status_checks": {"checks": []}})
        status, output = self.run_worker(api)
        self.assertEqual(status, 2)
        self.assertFalse(api.merged())

    def test_a_blocked_merge_reports_blocked_and_does_not_retry(self) -> None:
        api = FakeApi(merge_result=merge_module.MergeError("405 base branch policy"))
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertIn("blocked", output.lower())
        merges = [c for c in api.calls if "PUT" in c]
        self.assertEqual(len(merges), 1)

    def test_the_extra_validation_context_is_required_too(self) -> None:
        api = FakeApi(runs=[check_run(n) for n in REQUIRED])
        status, output = self.run_worker(api, extra_contexts=(VALIDATION_CONTEXT,))
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn(VALIDATION_CONTEXT, output)

    def test_the_extra_validation_context_passing_allows_the_merge(self) -> None:
        api = FakeApi(runs=[check_run(n) for n in (*REQUIRED, VALIDATION_CONTEXT)])
        status, output = self.run_worker(api, extra_contexts=(VALIDATION_CONTEXT,))
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())

    def test_a_forged_status_cannot_satisfy_the_extra_validation_context(self) -> None:
        """A collaborator can post a status; it carries no app id."""
        api = FakeApi(
            runs=[check_run(n) for n in REQUIRED],
            statuses={
                "state": "success",
                "statuses": [
                    {
                        "context": VALIDATION_CONTEXT,
                        "state": "success",
                        "created_at": "2026-09-08T21:00:00Z",
                    }
                ],
            },
        )
        status, output = self.run_worker(api, extra_contexts=(VALIDATION_CONTEXT,))
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn(VALIDATION_CONTEXT, output)

    # --- The workflow-file requirement, end to end ---

    def test_the_required_workflow_run_must_be_green_to_merge(self) -> None:
        api = FakeApi(workflow_runs=[workflow_run(conclusion="failure")])
        status, output = self.run_worker(api, required_workflow=VALIDATE_WORKFLOW)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn(VALIDATE_WORKFLOW, output)

    def test_a_green_required_workflow_run_allows_the_merge(self) -> None:
        api = FakeApi()
        status, output = self.run_worker(api, required_workflow=VALIDATE_WORKFLOW)
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())

    def test_a_named_context_does_not_substitute_for_the_workflow_run(self) -> None:
        """The check-run name is green; the workflow file never ran."""
        api = FakeApi(
            runs=[check_run(n) for n in (*REQUIRED, VALIDATION_CONTEXT)],
            workflow_runs=[],
        )
        status, output = self.run_worker(
            api,
            extra_contexts=(VALIDATION_CONTEXT,),
            required_workflow=VALIDATE_WORKFLOW,
        )
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())
        self.assertIn(VALIDATE_WORKFLOW, output)

    def test_no_workflow_is_consulted_when_none_is_required(self) -> None:
        api = FakeApi()
        self.run_worker(api)
        self.assertNotIn("/actions/workflows/", api.targets())

    # --- A transiently undecided mergeable flag is not a verdict ---

    def test_an_undecided_mergeable_flag_is_re_read_before_refusing(self) -> None:
        undecided = FakeApi.default_pull(mergeable=None, mergeable_state="unknown")
        api = FakeApi(
            pull_sequence=[undecided, undecided, FakeApi.default_pull()]
        )
        status, output = self.run_worker(api)
        self.assertEqual(status, 0, output)
        self.assertTrue(api.merged())

    def test_a_permanently_undecided_mergeable_flag_refuses(self) -> None:
        undecided = FakeApi.default_pull(mergeable=None, mergeable_state="unknown")
        api = FakeApi(pull=undecided)
        status, output = self.run_worker(api)
        self.assertEqual(status, 1)
        self.assertFalse(api.merged())


class CommandLineTests(unittest.TestCase):
    def test_the_base_ref_and_base_sha_are_required_arguments(self) -> None:
        parser = merge_module.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--repository", REPO, "--number", "7", "--head", HEAD]
            )

    def test_a_complete_invocation_parses(self) -> None:
        arguments = merge_module.build_parser().parse_args(
            [
                "--repository",
                REPO,
                "--number",
                "7",
                "--head",
                HEAD,
                "--base-ref",
                BASE_REF,
                "--base-sha",
                BASE,
                "--require-workflow",
                VALIDATE_WORKFLOW,
            ]
        )
        self.assertEqual(arguments.base_ref, BASE_REF)
        self.assertEqual(arguments.base_sha, BASE)
        self.assertEqual(arguments.require_workflow, VALIDATE_WORKFLOW)


if __name__ == "__main__":
    unittest.main()
