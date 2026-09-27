"""Fail-closed controls for exact-candidate Fast Tests reuse in expansion."""
from __future__ import annotations

import copy
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
from unittest import mock

import ci_expansion_fast_reuse as reuse
import ci_change_owned as owned


class FastReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "candidate_sha": "a" * 40,
            "config_digest": "b" * 64,
            "plan_digest": "c" * 64,
            "standing_targets": ["p::smoke", "p::featured", "q::smoke"],
            "package_expansion": ["p::smoke", "p::featured", "q::smoke"],
            "target_features": {
                "p::smoke": [], "p::featured": ["extra"], "q::smoke": [],
            },
            "manual_only_targets": [],
            "test_exclusions": [],
            "target_exclusions": [],
        }
        self.coverage = {
            "version": owned.STANDING_COVERAGE_VERSION,
            "candidate_sha": self.plan["candidate_sha"],
            "config_digest": self.plan["config_digest"],
            "execution": dict(owned.STANDING_EXECUTION),
            "selected_targets": list(self.plan["standing_targets"]),
            "executed_targets": list(self.plan["standing_targets"]),
            "selected_tests": [
                "p::smoke::works", "p::featured::works", "q::smoke::works"
            ],
            "executed_tests": [
                "p::smoke::works", "p::featured::works", "q::smoke::works"
            ],
            "success": True,
            "failures": [],
        }
        owned.attach_standing_coverage_digest(self.coverage)
        self.metadata = {
            "packages": [
                {"name": "p", "targets": [
                    {"name": "smoke", "kind": ["test"], "required-features": []},
                    {"name": "featured", "kind": ["test"], "required-features": ["extra"]},
                ]},
                {"name": "q", "targets": [
                    {"name": "smoke", "kind": ["test"], "required-features": []},
                ]},
            ]
        }
        self.junit = (
            b'<testsuites><testsuite>'
            b'<testcase classname="p::smoke" name="works"/>'
            b'<testcase classname="p::featured" name="works"/>'
            b'<testcase classname="q::smoke" name="works"/>'
            b'</testsuite></testsuites>'
        )

    def test_reuses_only_complete_same_configuration_targets(self) -> None:
        # A feature-bearing sibling shares p's expansion group. Neither p
        # target may be elided; the independent q target may be elided.
        result = reuse.authorized_targets(
            self.plan, self.coverage, self.junit, self.metadata,
            shard_targets=self.plan["package_expansion"],
            groups=[["p::smoke", "p::featured"], ["q::smoke"]],
        )
        self.assertEqual(result["reused_targets"], ["q::smoke"])
        self.assertEqual(result["reused_tests"], ["q::smoke::works"])
        self.assertEqual(len(result["fast_tests"]), 3)

    def test_rejects_stale_tampered_partial_failed_or_featured_receipt(self) -> None:
        cases = {
            "stale": lambda value: value.update(candidate_sha="d" * 40),
            "tampered digest": lambda value: value["executed_tests"].pop(),
            "partial": lambda value: (
                value["executed_tests"].pop(),
                owned.attach_standing_coverage_digest(value),
            ),
            "failed": lambda value: (
                value.update(success=False, failures=["failed"]),
                owned.attach_standing_coverage_digest(value),
            ),
            "different mode": lambda value: (
                value["execution"].update(run_ignored="all"),
                owned.attach_standing_coverage_digest(value),
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                coverage = copy.deepcopy(self.coverage)
                mutate(coverage)
                with self.assertRaises(ValueError):
                    reuse.authorized_targets(
                        self.plan, coverage, self.junit, self.metadata,
                        shard_targets=["q::smoke"], groups=[["q::smoke"]],
                    )

    def test_same_name_different_feature_and_partial_junit_run_full_target(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["packages"][1]["targets"][0]["required-features"] = ["special"]
        with self.assertRaises(ValueError):
            reuse.authorized_targets(
                self.plan, self.coverage, self.junit, metadata,
                shard_targets=["q::smoke"], groups=[["q::smoke"]],
            )
        with self.assertRaises(ValueError):
            reuse.authorized_targets(
                self.plan, self.coverage,
                self.junit.replace(b'<testcase classname="q::smoke" name="works"/>', b''),
                self.metadata,
                shard_targets=["q::smoke"], groups=[["q::smoke"]],
            )

    def test_ignored_and_exact_exclusion_targets_are_not_reused(self) -> None:
        for field, row in (
            ("manual_only_targets", {"identity": "q::smoke"}),
            ("test_exclusions", {"identity": "q::smoke::works"}),
        ):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan[field] = [row]
                result = reuse.authorized_targets(
                    plan, self.coverage, self.junit, self.metadata,
                    shard_targets=["q::smoke"], groups=[["q::smoke"]],
                )
                self.assertEqual(result["reused_targets"], [])
                self.assertEqual(result["fast_tests"], ["q::smoke::works"])

    def test_unavailable_artifact_yields_no_omission(self) -> None:
        result = reuse.fallback_manifest(self.plan, 0, "missing artifact")
        self.assertEqual(result["reused_targets"], [])
        self.assertEqual(result["reused_tests"], [])
        self.assertEqual(result["status"], "fallback")

    def test_source_requires_successful_exact_job_and_both_artifacts(self) -> None:
        repository = "Chelis-Lang/chelis"
        sha = self.plan["candidate_sha"]
        run_id = 42
        run = {"id": run_id, "head_sha": sha, "path": reuse.WORKFLOW,
               "event": "pull_request", "status": "completed",
               "head_repository": {"full_name": repository}}
        job = {"id": 7, "name": reuse.FAST_JOB, "conclusion": "success",
               "head_sha": sha}
        artifacts = [
            {"id": 8, "name": reuse.FAST_RECEIPTS, "expired": False,
             "workflow_run": {"id": run_id, "head_sha": sha}},
            {"id": 9, "name": reuse.FAST_JUNIT, "expired": False,
             "workflow_run": {"id": run_id, "head_sha": sha}},
        ]
        def archive(name, value):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as output:
                output.writestr(name, value)
            return buffer.getvalue()
        archives = {
            8: archive("ci-fast/coverage.json", json.dumps(self.coverage)),
            9: archive("junit.xml", self.junit),
        }
        def lookup(endpoint):
            if endpoint.endswith("/jobs?per_page=100"):
                return {"jobs": [job]}
            if endpoint.endswith("/artifacts?per_page=100"):
                return {"artifacts": artifacts}
            return {"workflow_runs": [run]}
        kwargs = {"api": lookup,
                  "download": lambda _repository, artifact_id: archives[artifact_id]}
        self.assertEqual(
            reuse._source_evidence(repository, sha, **kwargs)[0]["job_id"], 7
        )
        artifacts.pop()
        self.assertIsNone(reuse._source_evidence(repository, sha, **kwargs))
        artifacts.append({"id": 9, "name": reuse.FAST_JUNIT, "expired": False,
                          "workflow_run": {"id": run_id, "head_sha": sha}})
        job["conclusion"] = "failure"
        self.assertIsNone(reuse._source_evidence(repository, sha, **kwargs))
        job["conclusion"] = "success"
        run["head_sha"] = "d" * 40
        self.assertIsNone(reuse._source_evidence(repository, sha, **kwargs))

    def test_nextest_profile_settings_are_compared_not_just_names(self) -> None:
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "nextest.toml"
            config.write_text(
                '[profile.ci-fast]\ninherits="default"\nfail-fast=false\n'
                '[profile.ci-fast.junit]\npath="junit.xml"\n'
                '[profile.ci-full]\ninherits="default"\nfail-fast=false\n'
                '[profile.ci-full.junit]\npath="junit.xml"\n'
            )
            reuse.require_equivalent_profiles(config)
            config.write_text(config.read_text().replace(
                '[profile.ci-full]\ninherits="default"\nfail-fast=false',
                '[profile.ci-full]\ninherits="default"\nfail-fast=true',
            ))
            with self.assertRaisesRegex(ValueError, "profile settings differ"):
                reuse.require_equivalent_profiles(config)

    def test_old_candidate_keeps_legacy_expansion_interface(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            module = scripts / "ci_change_owned.py"
            module.write_text("RECEIPT_VERSION = 2\n")
            self.assertFalse(reuse.candidate_supports_reuse(root))
            (scripts / "ci_expansion_fast_reuse.py").write_text("# helper\n")
            self.assertFalse(reuse.candidate_supports_reuse(root))
            module.write_text("RECEIPT_VERSION = 3\n")
            self.assertTrue(reuse.candidate_supports_reuse(root))

    def test_manifest_digest_cannot_authorize_omission_by_itself(self) -> None:
        manifest = reuse.fallback_manifest(self.plan, 0, "missing artifact")
        manifest["reused_targets"] = ["q::smoke"]
        reuse._seal(manifest)
        with mock.patch.object(owned, "execution_shards", return_value={"0": ["q::smoke"]}):
            with self.assertRaisesRegex(ValueError, "do not exactly cover reused targets"):
                reuse.validate_manifest(self.plan, manifest, 0)

    def test_future_overlap_is_measured_and_never_self_authorized(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["standing_targets"].append("x::new")
        plan["package_expansion"].append("x::new")
        plan["target_features"]["x::new"] = []
        with self.assertRaises(ValueError):
            reuse.authorized_targets(
                plan, self.coverage, self.junit, self.metadata,
                shard_targets=["x::new"], groups=[["x::new"]],
            )


if __name__ == "__main__":
    unittest.main()
