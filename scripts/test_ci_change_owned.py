"""Change-owned CI planning, execution, schema, and receipt controls."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import ci_change_owned as owned


OWNER = {
    "workflow": "heavy-e2e.yml",
    "job": "full-workspace",
    "cadence": "daily 03:17 UTC and workflow_dispatch",
    "reason": "real build path is nightly-owned",
    "tracking_issue": "chelis#1824",
}


def package(
    name: str,
    targets: list[tuple[str, str, list[str]]],
    *,
    root: str | None = None,
    features: dict[str, list[str]] | None = None,
) -> dict:
    root = root or f"crates/{name}"
    return {
        "id": f"{name} 0.1.0 (path+file:///{root})",
        "name": name,
        "manifest_path": f"/repo/{root}/Cargo.toml",
        "features": features or {"default": []},
        "targets": [
            {
                "name": target,
                "kind": ["test"],
                "src_path": f"/repo/{src}",
                "required-features": required,
            }
            for target, src, required in targets
        ],
    }


def metadata(*packages: dict) -> dict:
    return {
        "workspace_root": "/repo",
        "workspace_members": [item["id"] for item in packages],
        "packages": list(packages),
    }


def config_text(
    *,
    standing: tuple[str, str] = ("p", "smoke"),
    target_exclusion: tuple[str, str] = ("p", "heavy"),
    test_exclusion: tuple[str, str, str] = ("p", "smoke", "slow_case"),
    path_rule: str = "scripts/",
) -> str:
    owner = "\n".join(f'{key} = {json.dumps(value)}' for key, value in OWNER.items())
    return f"""version = 2

[[standing_target]]
package = {json.dumps(standing[0])}
name = {json.dumps(standing[1])}

[[target_exclusion]]
package = {json.dumps(target_exclusion[0])}
name = {json.dumps(target_exclusion[1])}
{owner}

[[test_exclusion]]
package = {json.dumps(test_exclusion[0])}
target = {json.dumps(test_exclusion[1])}
name = {json.dumps(test_exclusion[2])}
{owner}

[[path_rule]]
prefix = {json.dumps(path_rule)}
disposition = "owner"
{owner}
"""


def fixture_metadata() -> dict:
    return metadata(
        package(
            "p",
            [
                ("smoke", "crates/p/tests/smoke.rs", []),
                ("heavy", "crates/p/tests/heavy.rs", []),
                ("gated", "crates/p/tests/gated.rs", ["extra"]),
                ("default_gated", "crates/p/tests/default_gated.rs", ["enabled"]),
            ],
            features={"default": ["enabled"], "enabled": [], "extra": []},
        ),
        package("q", [("smoke", "crates/q/tests/smoke.rs", [])]),
    )


def fixture_sources() -> dict[str, str]:
    return {
        "crates/p/tests/smoke.rs": (
            "#[test]\nfn fast_case() {}\n#[test]\nfn slow_case() {}\n"
        ),
        "crates/p/tests/heavy.rs": "#[test]\nfn heavy_case() {}\n",
        "crates/p/tests/gated.rs": "#[test]\nfn gated_case() {}\n",
        "crates/p/tests/default_gated.rs": "#[test]\nfn default_case() {}\n",
        "crates/q/tests/smoke.rs": "#[test]\nfn q_case() {}\n",
    }


def load_config(content: str | None = None) -> owned.Config:
    content = content or config_text()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        path.write_text(content)
        return owned.read_config(path)


class SchemaTests(unittest.TestCase):
    def test_strict_schema_reads_all_four_row_kinds(self) -> None:
        config = load_config()
        self.assertEqual(config.version, 2)
        self.assertEqual(config.standing_targets, (owned.Identity("p", "smoke"),))
        self.assertEqual(tuple(config.target_exclusions), (owned.Identity("p", "heavy"),))
        self.assertEqual(
            tuple(config.test_exclusions),
            (owned.TestIdentity("p", "smoke", "slow_case"),),
        )
        self.assertEqual(config.path_rules[0].prefix, "scripts/")

    def test_malformed_unknown_duplicate_and_ambiguous_rows_fail(self) -> None:
        valid = config_text()
        mutations = [
            valid.replace("version = 2", "version = 1", 1),
            valid + "\nunknown = true\n",
            valid + '\n[[standing_target]]\npackage = "p"\nname = "smoke"\n',
            valid + f'\n[[path_rule]]\nprefix = "scripts/"\ndisposition = "owner"\n'
            + "\n".join(f'{key} = {json.dumps(value)}' for key, value in OWNER.items())
            + "\n",
            valid + f'\n[[path_rule]]\nprefix = "scripts/sub/"\ndisposition = "owner"\n'
            + "\n".join(f'{key} = {json.dumps(value)}' for key, value in OWNER.items())
            + "\n",
            valid.replace('reason = "real build path is nightly-owned"', "reason = \"\"", 1),
            valid.replace(
                'tracking_issue = "chelis#1824"',
                'tracking_issue = "PR-126"',
                1,
            ),
            valid.replace('name = "smoke"', 'name = "../smoke"', 1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation[-100:]), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(mutation)
                with self.assertRaises(ValueError):
                    owned.read_config(path)

    def test_stale_package_target_test_and_path_rule_fail_closed(self) -> None:
        config = load_config()
        tracked = set(fixture_sources()) | {"scripts/tool.py"}
        cases = [
            (config_text(standing=("missing", "smoke")), "package"),
            (config_text(standing=("p", "missing")), "target"),
            (config_text(test_exclusion=("p", "smoke", "missing_case")), "test"),
            (config_text(path_rule="missing/"), "path rule"),
        ]
        for content, message in cases:
            with self.subTest(message=message):
                candidate = load_config(content)
                with self.assertRaisesRegex(ValueError, message):
                    owned.validate_config(
                        candidate,
                        fixture_metadata(),
                        tracked,
                        fixture_sources().__getitem__,
                    )
        owned.validate_config(
            config, fixture_metadata(), tracked, fixture_sources().__getitem__
        )

    def test_deleted_exact_path_rule_is_live_from_base_inventory(self) -> None:
        config = load_config(config_text(path_rule="removed-root.toml"))
        with self.assertRaisesRegex(ValueError, "path rule"):
            owned.validate_config(
                config,
                fixture_metadata(),
                set(fixture_sources()),
                fixture_sources().__getitem__,
            )
        owned.validate_config(
            config,
            fixture_metadata(),
            set(fixture_sources()) | {"removed-root.toml"},
            fixture_sources().__getitem__,
        )

    def test_target_and_test_exclusions_cannot_contradict_other_rows(self) -> None:
        standing_excluded = config_text(target_exclusion=("p", "smoke"))
        test_under_excluded = config_text(
            target_exclusion=("p", "heavy"),
            test_exclusion=("p", "heavy", "heavy_case"),
        )
        for content in (standing_excluded, test_under_excluded):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    load_config(content)

    def test_repository_manifest_has_exact_selected_inventory_and_owners(self) -> None:
        config = owned.read_config(
            Path(__file__).resolve().parents[1] / ".config/ci-test-targets.toml"
        )
        self.assertTrue({
            owned.Identity("chelis-types", "expand_insert_dispatch_family"),
            owned.Identity("chelis-types", "issue_1294_standard_lowerings"),
        } <= set(config.standing_targets))
        self.assertEqual(len(config.target_exclusions), 3)
        self.assertEqual(len(config.test_exclusions), 6)
        for owner in (
            *config.target_exclusions.values(),
            *config.test_exclusions.values(),
        ):
            self.assertEqual(owner.workflow, "heavy-e2e.yml")
            self.assertEqual(owner.job, "full-workspace")
            self.assertEqual(owner.cadence, "daily 03:17 UTC and workflow_dispatch")
        self.assertEqual(
            sum(
                owner.tracking_issue == "chelis#1824"
                for owner in (
                    *config.target_exclusions.values(),
                    *config.test_exclusions.values(),
                )
            ),
            8,
        )
        recursive = owned.TestIdentity(
            "chelis-cli",
            "issue_1293_redteam_round4",
            "recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c",
        )
        self.assertEqual(
            config.test_exclusions[recursive].tracking_issue,
            "chelis#1293",
        )
        broad_unsupported = {
            "examples/",
            "tests/",
            "scripts/",
            ".github/",
            ".config/",
            "grammars/",
            "editors/",
            "nix/",
            "devenv/",
            "benchmarks/",
        }
        self.assertFalse(
            broad_unsupported & {rule.prefix for rule in config.path_rules}
        )
        root = Path(__file__).resolve().parents[1]
        heavy = (root / ".github/workflows/heavy-e2e.yml").read_text()
        self.assertIn("\n  full-workspace:\n", heavy)
        self.assertIn('cron: "17 3 * * *"', heavy)
        for rule in config.path_rules:
            if rule.owner is not None:
                workflow = (root / ".github/workflows" / rule.owner.workflow).read_text()
                self.assertIn(f"\n  {rule.owner.job}:\n", workflow)
        by_path = {rule.prefix: rule for rule in config.path_rules}
        self.assertEqual(by_path["scripts/test_nextest_profile_partition.py"].owner.job,
                         "full-workspace")

    def test_canonical_release_shared_pins_have_required_gate_owners(self) -> None:
        from scripts import bump_compiler_pins as bump

        root = Path(__file__).resolve().parents[1]
        config = owned.read_config(root / ".config/ci-test-targets.toml")
        paths = [*bump.PINNED_REAL_TOML_FILES,
                 *(path / "reef.lock" for path in bump.PINNED_REAL_LOCK_DIRS)]
        example_paths = {str(path.relative_to(root)) for path in paths
                         if path.is_relative_to(root / "examples")}
        self.assertEqual(len(example_paths), 5)
        for path in sorted(example_paths):
            with self.subTest(path=path):
                rules = [rule for rule in config.path_rules if rule.matches(path)]
                self.assertEqual(len(rules), 1, path)
                rule = rules[0]
                self.assertEqual(rule.prefix, path)
                self.assertEqual(rule.disposition, "owner")
                self.assertEqual((rule.owner.workflow, rule.owner.job), ("ci.yml", "ci-fast"))
        self.assertIn(owned.Identity("chelis-cli", "compiler_pin_tripwire"), config.standing_targets)
        hull = str(bump.HULL_MANIFEST.relative_to(root))
        rules = [rule for rule in config.path_rules if rule.matches(hull)]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].prefix, hull)
        self.assertEqual((rules[0].owner.workflow, rules[0].owner.job),
                         ("conformance.yml", "conformance"))
        # Neighboring unreviewed sources must not inherit release-pin authority.
        for path in ("examples/illustrative/phase3g_text_pipeline/new.ch",
                     "examples/nautilus_quantile_contract/new.ch",
                     "tests/conformance/hull/new.json"):
            self.assertFalse(any(rule.matches(path) for rule in config.path_rules), path)
            self.assertFalse(owned.is_docs_only([path]))

    def test_release_workflow_has_a_required_script_unit_owner(self) -> None:
        # chelis#2027 precedent: a canonical release input outside every Cargo
        # package root needs an exact owner rule, or the planner fails closed on
        # every release-workflow edit. The owner claim is evidence-backed: each
        # named script-unit suite text-parses release.yml on every pull request.
        root = Path(__file__).resolve().parents[1]
        config = owned.read_config(root / ".config/ci-test-targets.toml")
        release = ".github/workflows/release.yml"
        rules = [rule for rule in config.path_rules if rule.matches(release)]
        self.assertEqual(len(rules), 1, release)
        rule = rules[0]
        self.assertEqual(rule.prefix, release)
        self.assertEqual(rule.disposition, "owner")
        self.assertEqual((rule.owner.workflow, rule.owner.job), ("ci.yml", "script-unit"))
        self.assertFalse(owned.is_docs_only([release]))
        for suite in ("test_changelog.py",
                      "test_release_workflow_pyo3_isolation.py",
                      "test_release_runtime_header_manifest.py",
                      "test_ci_cvc5_build.py",
                      "test_glibc231_workflows.py",
                      "test_installed_artifact_canary.py"):
            with self.subTest(suite=suite):
                self.assertIn("release.yml", (root / "scripts" / suite).read_text(encoding="utf-8"))
        # Sibling workflow files and release helper scripts must not inherit
        # the exact rule; an unreviewed lane still needs its own mapping.
        for path in (".github/workflows/new-release-lane.yml",
                     ".github/workflows/heavy-e2e.yml",
                     ".github/scripts/verify_release_smt.py"):
            self.assertFalse(any(rule.matches(path) for rule in config.path_rules), path)
            self.assertFalse(owned.is_docs_only([path]))

    def test_path_rules_cannot_override_existing_docs_only_policy(self) -> None:
        for path in ("README.md", "spec/05-risc-primitives.md", "new-tools/new.py"):
            text = config_text() + (
                f'\n[[path_rule]]\nprefix = "{path}"\ndisposition = "docs_only"\n'
            )
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "disposition"):
                load_config(text)


class MetadataAndDiffTests(unittest.TestCase):
    def test_default_feature_eligibility_and_duplicate_names_are_package_qualified(self) -> None:
        targets = owned.integration_targets(fixture_metadata())
        self.assertIn(owned.Identity("p", "default_gated"), targets)
        self.assertNotIn(owned.Identity("p", "gated"), targets)
        self.assertIn(owned.Identity("p", "smoke"), targets)
        self.assertIn(owned.Identity("q", "smoke"), targets)
        self.assertNotEqual(
            targets[owned.Identity("p", "smoke")].src_path,
            targets[owned.Identity("q", "smoke")].src_path,
        )
        self.assertEqual(
            owned.TestIdentity.parse("p::smoke::nested::case").test,
            "nested::case",
        )

    def test_rename_diff_is_parsed_as_delete_and_add(self) -> None:
        raw = (
            b"M\0crates/p/src/lib.rs\0"
            b"R100\0crates/p/tests/old.rs\0crates/p/tests/new.rs\0"
            b"D\0removed.txt\0"
        )
        records = owned.parse_name_status_z(raw)
        self.assertEqual(
            [(record.status, record.old_path, record.path) for record in records],
            [
                ("M", None, "crates/p/src/lib.rs"),
                ("R100", "crates/p/tests/old.rs", "crates/p/tests/new.rs"),
                ("D", None, "removed.txt"),
            ],
        )
        self.assertEqual(
            owned.diff_paths(records),
            {"crates/p/src/lib.rs", "crates/p/tests/old.rs", "crates/p/tests/new.rs", "removed.txt"},
        )

    def test_malformed_nul_diff_is_rejected(self) -> None:
        for raw in (
            b"M\0path",
            b"R100\0old\0",
            b"C100\0old\0copy\0",
            b"R101\0old\0new\0",
            b"Rxx\0old\0new\0",
            b"U\0path\0",
            b"X\0path\0",
            b"\0path\0",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                owned.parse_name_status_z(raw)


class PlanningTests(unittest.TestCase):
    def plan(
        self,
        records: list[owned.ChangeRecord],
        *,
        base: dict | None = None,
        candidate: dict | None = None,
        tracked: set[str] | None = None,
    ) -> dict:
        sources = fixture_sources()
        return owned.make_plan(
            mode="push",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            records=records,
            base_metadata=base or fixture_metadata(),
            candidate_metadata=candidate or fixture_metadata(),
            config=load_config(),
            tracked_paths=tracked or (set(sources) | {"scripts/tool.py"}),
            source_reader=sources.__getitem__,
        )

    def test_direct_target_and_package_expansion_are_disjoint(self) -> None:
        plan = self.plan(
            [
                owned.ChangeRecord("M", "crates/p/tests/smoke.rs"),
                owned.ChangeRecord("M", "crates/p/src/lib.rs"),
            ]
        )
        self.assertEqual(plan["change_owned"], ["p::smoke"])
        self.assertIn("p::default_gated", plan["package_expansion"])
        self.assertNotIn("p::smoke", plan["package_expansion"])
        self.assertNotIn("p::heavy", plan["package_expansion"])
        self.assertEqual(plan["selected_packages"], ["p"])
        owned.verify_plan_digest(plan)

    def test_mixed_code_and_new_prose_use_the_existing_docs_only_disposition(self) -> None:
        for path in ("changelog.d/new-fix.fixed.md", "docs/new-page.md",
                     "spec/design/new-assessment.md", "openspec/changes/new/.openspec.yaml"):
            with self.subTest(path=path):
                plan = self.plan([owned.ChangeRecord("M", "crates/p/tests/smoke.rs"),
                                  owned.ChangeRecord("A", path)])
                self.assertEqual(plan["change_owned"], ["p::smoke"])
                self.assertEqual(plan["path_dispositions"][1],
                                 {"path": path, "status": "A", "kind": "docs_only"})

    def test_executable_docs_and_unknown_code_still_require_a_reviewed_mapping(self) -> None:
        for path in ("spec/05-risc-primitives.md", "docs/investigations/remediation_status_2026_08_04.md",
                     "openspec/config.yaml", "new-tools/check.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "unclassified changed path"):
                self.plan([owned.ChangeRecord("M", "crates/p/tests/smoke.rs"),
                           owned.ChangeRecord("A", path)])
    def test_added_target_is_owned_and_renamed_target_is_delete_plus_add(self) -> None:
        base = metadata(
            package("p", [("old", "crates/p/tests/old.rs", []), ("heavy", "crates/p/tests/heavy.rs", [])]),
            package("q", [("smoke", "crates/q/tests/smoke.rs", [])]),
        )
        candidate = metadata(
            package("p", [("new", "crates/p/tests/new.rs", []), ("heavy", "crates/p/tests/heavy.rs", [])]),
            package("q", [("smoke", "crates/q/tests/smoke.rs", [])]),
        )
        content = config_text(standing=("q", "smoke"), target_exclusion=("p", "heavy"),
                              test_exclusion=("q", "smoke", "q_case"))
        config = load_config(content)
        sources = {
            "crates/p/tests/new.rs": "#[test]\nfn new_case() {}\n",
            "crates/p/tests/heavy.rs": "#[test]\nfn heavy_case() {}\n",
            "crates/q/tests/smoke.rs": "#[test]\nfn q_case() {}\n",
        }
        plan = owned.make_plan(
            mode="push",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            records=[owned.ChangeRecord("R100", "crates/p/tests/new.rs", "crates/p/tests/old.rs")],
            base_metadata=base,
            candidate_metadata=candidate,
            config=config,
            tracked_paths=set(sources) | {"scripts/tool.py"},
            source_reader=sources.__getitem__,
        )
        self.assertEqual(plan["change_owned"], ["p::new"])
        disposition = {row["path"]: row for row in plan["path_dispositions"]}
        self.assertEqual(disposition["crates/p/tests/old.rs"]["kind"], "integration_target_deleted")
        self.assertEqual(disposition["crates/p/tests/new.rs"]["kind"], "integration_target_added")

    def test_manifest_only_added_target_is_change_owned(self) -> None:
        base = fixture_metadata()
        candidate = metadata(
            package(
                "p",
                [
                    ("smoke", "crates/p/tests/smoke.rs", []),
                    ("heavy", "crates/p/tests/heavy.rs", []),
                    ("gated", "crates/p/tests/gated.rs", ["extra"]),
                    (
                        "default_gated",
                        "crates/p/tests/default_gated.rs",
                        ["enabled"],
                    ),
                    ("new", "crates/p/tests/existing.rs", []),
                ],
                features={"default": ["enabled"], "enabled": [], "extra": []},
            ),
            package("q", [("smoke", "crates/q/tests/smoke.rs", [])]),
        )
        sources = fixture_sources() | {
            "crates/p/tests/existing.rs": "#[test]\nfn existing_case() {}\n"
        }
        plan = owned.make_plan(
            mode="push",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            records=[owned.ChangeRecord("M", "crates/p/Cargo.toml")],
            base_metadata=base,
            candidate_metadata=candidate,
            config=load_config(),
            tracked_paths=set(sources) | {"scripts/tool.py", "crates/p/Cargo.toml"},
            source_reader=sources.__getitem__,
        )
        self.assertIn("p::new", plan["change_owned"])
        self.assertIn(
            {
                "identity": "p::new",
                "kind": "integration_target_added",
                "src_path": "crates/p/tests/existing.rs",
            },
            plan["target_dispositions"],
        )

    def test_unknown_path_and_ambiguous_package_roots_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unclassified changed path"):
            self.plan([owned.ChangeRecord("A", "brand-new-root/file.txt")])
        nested = metadata(
            package(
                "outer",
                [
                    ("one", "nested/tests/one.rs", []),
                    ("heavy", "nested/tests/heavy.rs", []),
                ],
                root="nested",
            ),
            package(
                "inner",
                [
                    ("two", "nested/inner/tests/two.rs", []),
                    ("smoke", "nested/inner/tests/smoke.rs", []),
                ],
                root="nested/inner",
            ),
        )
        nested_config = load_config(
            config_text(
                standing=("outer", "one"),
                target_exclusion=("outer", "heavy"),
                test_exclusion=("inner", "two", "two_case"),
            )
        )
        nested_sources = {
            "nested/tests/one.rs": "#[test]\nfn one_case() {}\n",
            "nested/tests/heavy.rs": "#[test]\nfn heavy_case() {}\n",
            "nested/inner/tests/two.rs": "#[test]\nfn two_case() {}\n",
            "nested/inner/tests/smoke.rs": "#[test]\nfn smoke_case() {}\n",
            "scripts/tool.py": "",
        }
        with self.assertRaisesRegex(ValueError, "ambiguous package path"):
            owned.make_plan(
                mode="push",
                base_sha="a" * 40,
                candidate_sha="b" * 40,
                records=[owned.ChangeRecord("M", "nested/inner/src/lib.rs")],
                base_metadata=nested,
                candidate_metadata=nested,
                config=nested_config,
                tracked_paths=set(nested_sources),
                source_reader=nested_sources.__getitem__,
            )

    def test_path_owner_and_package_rules_have_exact_dispositions(self) -> None:
        package_owner = config_text().replace(
            'prefix = "scripts/"\ndisposition = "owner"\n'
            + "\n".join(f'{key} = {json.dumps(value)}' for key, value in OWNER.items()),
            'prefix = "scripts/"\ndisposition = "packages"\npackages = ["p", "q"]',
        )
        config = load_config(package_owner)
        sources = fixture_sources()
        plan = owned.make_plan(
            mode="push",
            base_sha="a" * 40,
            candidate_sha="b" * 40,
            records=[owned.ChangeRecord("M", "scripts/tool.py")],
            base_metadata=fixture_metadata(),
            candidate_metadata=fixture_metadata(),
            config=config,
            tracked_paths=set(sources) | {"scripts/tool.py"},
            source_reader=sources.__getitem__,
        )
        self.assertEqual(plan["selected_packages"], ["p", "q"])
        self.assertEqual(plan["path_dispositions"][0]["kind"], "path_rule_packages")

    def test_excluded_direct_target_resolves_to_alternative_owner(self) -> None:
        plan = self.plan([owned.ChangeRecord("M", "crates/p/tests/heavy.rs")])
        self.assertEqual(plan["change_owned"], [])
        disposition = plan["path_dispositions"][0]
        self.assertEqual(disposition["kind"], "integration_target_excluded")
        self.assertEqual(disposition["owner"]["workflow"], "heavy-e2e.yml")

    def test_plan_digest_detects_mutation(self) -> None:
        plan = self.plan([owned.ChangeRecord("M", "crates/p/tests/smoke.rs")])
        mutated = copy.deepcopy(plan)
        mutated["change_owned"] = []
        with self.assertRaises(ValueError):
            owned.verify_plan_digest(mutated)

    def test_pr_candidate_requires_two_parents_and_exact_event_head(self) -> None:
        with mock.patch.object(owned, "git_output") as git_output:
            git_output.return_value = b"merge base head extra\n"
            with self.assertRaisesRegex(ValueError, "exactly two parents"):
                owned.resolve_pr_commits(Path("/repo"), "merge", "head")
            git_output.return_value = b"merge base other\n"
            with self.assertRaisesRegex(ValueError, "event pull-request head"):
                owned.resolve_pr_commits(Path("/repo"), "merge", "head")
            git_output.return_value = b"merge base head\n"
            self.assertEqual(
                owned.resolve_pr_commits(Path("/repo"), "merge", "head"),
                ("base", "merge"),
            )


class ShardingAndExecutionTests(unittest.TestCase):
    def test_expansion_deadline_preserves_receipts_at_every_command_boundary(self) -> None:
        identity = owned.Identity("p", "smoke")
        shard = owned.shard_for(identity)
        later = next(
            owned.Identity("p", f"z{n}") for n in range(100)
            if owned.shard_for(owned.Identity("p", f"z{n}")) == shard
        )
        for expired_call, truncated_junit in (
            (0, False), (1, False), (2, False), (3, False),
            (4, False), (4, True), (None, False),
        ):
            with self.subTest(expired_call=expired_call, truncated_junit=truncated_junit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / "target"
                plan = self._plan(lane="package-expansion")
                plan["eligible_targets"].append(later.canonical)
                plan["package_expansion"].append(later.canonical)
                plan["shards"]["package_expansion"] = owned.shard_map([identity, later])
                owned.attach_plan_digest(plan)
                calls = []

                def run(command, **kwargs):
                    index = len(calls)
                    calls.append((command, kwargs))
                    if index == expired_call:
                        if truncated_junit:
                            junit = target / "nextest/ci-full/junit.xml"
                            junit.parent.mkdir(parents=True, exist_ok=True)
                            junit.write_text('<testsuite><testcase name="fast_case"/>')
                        raise subprocess.TimeoutExpired(
                            command, kwargs.get("timeout", 0),
                            output=b"partial stdout", stderr=b"partial stderr",
                        )
                    if command[1] == "build":
                        return subprocess.CompletedProcess(command, 0, "", "")
                    name = command[command.index("--test") + 1]
                    if command[2] == "list":
                        cases = {"fast_case": {"ignored": False, "filter-match": {"status": "matches"}}}
                        if name == "smoke":
                            cases["slow_case"] = {"ignored": False, "filter-match": {"status": "mismatch"}}
                        payload = {"rust-suites": {f"p::{name}": {"testcases": cases}}}
                        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
                    junit = target / "nextest/ci-full/junit.xml"
                    junit.parent.mkdir(parents=True, exist_ok=True)
                    junit.write_text('<testsuite><testcase name="fast_case"/></testsuite>')
                    return subprocess.CompletedProcess(command, 0, "", "")

                with (
                    mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target)}),
                    mock.patch.object(owned, "_commit", return_value="b" * 40),
                ):
                    receipt = owned.execute_shard(
                        plan, lane="package-expansion", shard=shard,
                        output=root / "receipt", repo=root, runner=run,
                    )
                self.assertEqual(owned.load_receipts(root / "receipt"), [receipt])
                self.assertEqual(receipt["selected_targets"], [identity.canonical, later.canonical])
                self.assertEqual(receipt["success"], expired_call is None)
                self.assertEqual(len(calls), 5 if expired_call is None else expired_call + 1)
                timeouts = [kwargs["timeout"] for _, kwargs in calls]
                self.assertTrue(all(0 < value <= 960 for value in timeouts))
                self.assertEqual(timeouts, sorted(timeouts, reverse=True))
                if expired_call is not None:
                    self.assertIn("execution deadline", " ".join(receipt["failures"]))
                    commands = json.loads((root / "receipt/commands.json").read_text())
                    self.assertEqual(commands[-1]["stdout"], "partial stdout")
                    self.assertEqual(commands[-1]["stderr"], "partial stderr")
                    summary = owned.summarize_package_expansion(plan, [receipt])
                    self.assertFalse(summary["observed_success"])
                if expired_call in (3, 4):
                    self.assertEqual(receipt["executed_tests"], ["p::smoke::fast_case"])
                if truncated_junit:
                    self.assertIn("malformed JUnit", " ".join(receipt["failures"]))
                    self.assertEqual(owned._junit_tests(root / "receipt/junit.xml", identity), ["p::smoke::fast_case"])

    def test_budget_is_shared_and_no_command_starts_after_it_expires(self) -> None:
        plan = self._plan(lane="package-expansion")
        current_time = [100.0]
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            current_time[0] += 961
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp, (
            mock.patch.object(owned, "_commit", return_value="b" * 40)
        ), mock.patch.object(owned.time, "monotonic", side_effect=lambda: current_time[0]):
            receipt = owned.execute_shard(
                plan, lane="package-expansion",
                shard=owned.shard_for(owned.Identity("p", "smoke")),
                output=Path(tmp), repo=Path(tmp), runner=run,
            )
        self.assertEqual(len(calls), 1)
        self.assertFalse(receipt["success"])
        self.assertTrue(receipt["soft_budget_exceeded"])
        self.assertEqual(receipt["executed_targets"], [])
        self.assertIn("execution deadline", " ".join(receipt["failures"]))

    def test_expiry_after_listing_does_not_claim_the_target_executed(self) -> None:
        plan = self._plan(lane="package-expansion")
        current_time = [100.0]
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[1] == "build":
                return subprocess.CompletedProcess(command, 0, "", "")
            current_time[0] += 961
            payload = {"rust-suites": {"p::smoke": {"testcases": {
                "fast_case": {"ignored": False, "filter-match": {"status": "matches"}},
                "slow_case": {"ignored": False, "filter-match": {"status": "mismatch"}},
            }}}}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with tempfile.TemporaryDirectory() as tmp, (
            mock.patch.object(owned, "_commit", return_value="b" * 40)
        ), mock.patch.object(owned.time, "monotonic", side_effect=lambda: current_time[0]):
            receipt = owned.execute_shard(
                plan, lane="package-expansion",
                shard=owned.shard_for(owned.Identity("p", "smoke")),
                output=Path(tmp), repo=Path(tmp), runner=run,
            )
        self.assertEqual(len(calls), 2)
        self.assertFalse(receipt["success"])
        self.assertEqual(receipt["selected_tests"], ["p::smoke::fast_case"])
        self.assertEqual(receipt["executed_targets"], [])
        self.assertEqual(receipt["executed_tests"], [])

    def test_shards_are_deterministic_package_qualified_and_cover_all(self) -> None:
        identities = [
            owned.Identity("p", "smoke"),
            owned.Identity("q", "smoke"),
            owned.Identity("p", "default_gated"),
        ]
        first = owned.shard_map(identities)
        second = owned.shard_map(reversed(identities))
        self.assertEqual(first, second)
        self.assertEqual(
            sorted(identity for rows in first.values() for identity in rows),
            sorted(identity.canonical for identity in identities),
        )
        self.assertNotEqual(
            owned.shard_for(owned.Identity("p", "smoke")),
            owned.shard_for(owned.Identity("other-package", "smoke")),
        )

    def test_commands_are_package_scoped_and_apply_only_exact_test_exclusions(self) -> None:
        config = load_config()
        p = owned.Identity("p", "smoke")
        q = owned.Identity("q", "smoke")
        p_command = owned.target_command(p, config.test_exclusions, list_only=False)
        q_command = owned.target_command(q, config.test_exclusions, list_only=False)
        self.assertEqual(p_command[p_command.index("-p") + 1], "p")
        self.assertEqual(q_command[q_command.index("-p") + 1], "q")
        self.assertEqual(p_command[p_command.index("--test") + 1], "smoke")
        self.assertIn("-E", p_command)
        self.assertIn(
            "test(/^slow_case$/)",
            p_command[p_command.index("-E") + 1],
        )
        self.assertIn("--locked", p_command)
        self.assertNotIn("-E", q_command)

    def _plan(self, *, lane: str = "change-owned") -> dict:
        lane_key = owned.LANE_KEYS[lane]
        identity = owned.Identity("p", "smoke")
        plan = {
            "version": 1,
            "mode": "push",
            "base_sha": "a" * 40,
            "candidate_sha": "b" * 40,
            "event_pr_head": None,
            "changed_records": [],
            "path_dispositions": [],
            "target_dispositions": [],
            "selected_packages": ["p"],
            "eligible_targets": [identity.canonical],
            "change_owned": (
                [identity.canonical] if lane_key == "change_owned" else []
            ),
            "package_expansion": (
                [identity.canonical] if lane_key == "package_expansion" else []
            ),
            "standing_targets": [],
            "target_exclusions": [],
            "test_exclusions": [
                {"identity": "p::smoke::slow_case", "owner": OWNER}
            ],
            "shards": {
                "change_owned": owned.shard_map(
                    [identity] if lane_key == "change_owned" else []
                ),
                "package_expansion": owned.shard_map(
                    [identity] if lane_key == "package_expansion" else []
                ),
            },
        }
        owned.attach_plan_digest(plan)
        return plan

    def test_listing_rejects_unconfigured_filtering_and_stale_exclusions(self) -> None:
        identity = owned.Identity("p", "smoke")
        listing = {
            "rust-suites": {
                "p::smoke": {
                    "testcases": {
                        "fast_case": {
                            "ignored": False,
                            "filter-match": {"status": "matches"},
                        },
                        "slow_case": {
                            "ignored": False,
                            "filter-match": {"status": "mismatch"},
                        },
                    }
                }
            }
        }
        exclusions = {owned.TestIdentity("p", "smoke", "slow_case"): OWNER}
        self.assertEqual(
            owned._listing_tests(listing, identity, exclusions),
            ["p::smoke::fast_case"],
        )
        listing["rust-suites"]["p::smoke"]["testcases"]["hidden_case"] = {
            "ignored": False,
            "filter-match": {"status": "mismatch"},
        }
        with self.assertRaisesRegex(ValueError, "nonmatching active tests"):
            owned._listing_tests(listing, identity, exclusions)
        del listing["rust-suites"]["p::smoke"]["testcases"]["hidden_case"]
        del listing["rust-suites"]["p::smoke"]["testcases"]["slow_case"]
        with self.assertRaisesRegex(ValueError, "nonmatching active tests"):
            owned._listing_tests(listing, identity, exclusions)

    def test_listing_rejects_malformed_filter_match_shape(self) -> None:
        identity = owned.Identity("p", "smoke")
        listing = {
            "rust-suites": {
                "p::smoke": {
                    "testcases": {
                        "fast_case": {
                            "ignored": False,
                            "filter-match": "matches",
                        }
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "malformed filter match"):
            owned._listing_tests(listing, identity, {})

    def test_nonempty_shard_builds_products_once_before_target_commands(self) -> None:
        plan = self._plan()
        identity = owned.Identity("p", "smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            output = root / "receipt"
            calls: list[list[str]] = []

            def run(command, **kwargs):
                self.assertNotIn("timeout", kwargs)
                calls.append(command)
                if command[1:3] == ["nextest", "list"]:
                    payload = {
                        "rust-suites": {
                            identity.canonical: {
                                "testcases": {
                                    "fast_case": {
                                        "ignored": False,
                                        "filter-match": {"status": "matches"},
                                    },
                                    "slow_case": {
                                        "ignored": False,
                                        "filter-match": {"status": "mismatch"},
                                    },
                                }
                            }
                        }
                    }
                    return mock.Mock(
                        stdout=json.dumps(payload), stderr="", returncode=0
                    )
                if command[1:3] == ["nextest", "run"]:
                    junit = target / "nextest/ci-full/junit.xml"
                    junit.parent.mkdir(parents=True, exist_ok=True)
                    junit.write_text(
                        '<testsuite><testcase name="fast_case"/></testsuite>'
                    )
                return mock.Mock(stdout="", stderr="", returncode=0)

            with (
                mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target)}),
                mock.patch.object(owned, "_commit", return_value="b" * 40),
            ):
                receipt = owned.execute_shard(
                    plan,
                    lane="change-owned",
                    shard=owned.shard_for(identity),
                    output=output,
                    repo=root,
                    runner=run,
                )
            self.assertTrue(receipt["success"])
            self.assertEqual(
                calls[0],
                ["cargo", "build", "--workspace", "--lib", "--bins", "--locked"],
            )
            self.assertEqual(sum(command[1] == "build" for command in calls), 1)
            self.assertEqual(receipt["executed_tests"], ["p::smoke::fast_case"])
            self.assertEqual(
                set(receipt["sidecars"]),
                {"commands.json", "timings.json", "test-list.json", "junit.xml"},
            )
            self.assertEqual(len(owned.load_receipts(output)), 1)

    def test_product_build_failure_writes_receipt_and_skips_targets(self) -> None:
        plan = self._plan()
        identity = owned.Identity("p", "smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "receipt"
            calls: list[list[str]] = []

            def run(command, **kwargs):
                calls.append(command)
                raise subprocess.CalledProcessError(
                    1, command, output="", stderr="build failed"
                )

            with mock.patch.object(owned, "_commit", return_value="b" * 40):
                receipt = owned.execute_shard(
                    plan,
                    lane="change-owned",
                    shard=owned.shard_for(identity),
                    output=output,
                    repo=root,
                    runner=run,
                )
            self.assertFalse(receipt["success"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], "build")
            self.assertTrue((output / "receipt.json").is_file())

    def test_candidate_head_mismatch_stops_before_cargo(self) -> None:
        runner = mock.Mock()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(owned, "_commit", return_value="c" * 40),
            self.assertRaisesRegex(ValueError, "checked-out HEAD"),
        ):
            owned.execute_shard(
                self._plan(),
                lane="change-owned",
                shard=0,
                output=Path(tmp),
                runner=runner,
            )
        runner.assert_not_called()

    def test_run_shard_returns_zero_after_captured_failure_receipt(self) -> None:
        plan = self._plan()
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_bytes(owned.canonical_json(plan))
            with mock.patch.object(
                owned,
                "execute_shard",
                return_value={"success": False},
            ):
                result = owned.main(
                    [
                        "run-shard",
                        "--plan",
                        str(plan_path),
                        "--lane",
                        "change-owned",
                        "--shard",
                        "0",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
        self.assertEqual(result, 0)

    def test_cli_spellings_match_the_workflow_contract(self) -> None:
        parser = owned.build_parser()
        plan = parser.parse_args(
            [
                "plan",
                "--event-name",
                "push",
                "--pr-head",
                "a" * 40,
                "--before",
                "b" * 40,
                "--after",
                "c" * 40,
                "--output",
                "plan.json",
            ]
        )
        self.assertEqual(plan.command, "plan")
        shard = parser.parse_args(
            [
                "run-shard",
                "--plan",
                "plan.json",
                "--lane",
                "package-expansion",
                "--shard",
                "3",
                "--output",
                "receipt",
            ]
        )
        self.assertEqual((shard.lane, shard.shard), ("package-expansion", 3))
        report = parser.parse_args(
            [
                "report",
                "--plan",
                "plan.json",
                "--lane",
                "change-owned",
                "--receipts-root",
                "receipts",
                "--output",
                "report",
                "--required",
            ]
        )
        self.assertTrue(report.required)


@unittest.skipUnless(os.name == "posix", "hosted expansion uses POSIX process groups")
class BoundedCommandTests(unittest.TestCase):
    def test_interruption_cleans_up_the_owned_process_group(self) -> None:
        process = mock.Mock(pid=4321)
        process.communicate.side_effect = [KeyboardInterrupt(), ("out", "err")]
        manager = mock.MagicMock()
        manager.__enter__.return_value = process
        with (
            mock.patch.object(owned.subprocess, "Popen", return_value=manager),
            mock.patch.object(owned.os, "killpg") as kill,
            self.assertRaises(KeyboardInterrupt),
        ):
            owned.run_command(["cargo", "build"], timeout=1, check=True)
        kill.assert_called_once_with(4321, owned.signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)

    def test_real_executor_timeout_writes_a_verifiable_unsuccessful_receipt(self) -> None:
        plan = ShardingAndExecutionTests()._plan(lane="package-expansion")
        for stage in ("build", "list", "run"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # Exercise the production process runner and serializer together,
                # with an executable stand-in for Cargo instead of a real build.
                cargo = root / "cargo"
                cargo.write_text(
                    f"#!{sys.executable}\n"
                    "import json,sys,time\n"
                    f"if {stage!r} in sys.argv[1:3]:\n"
                    " print('deadline output', flush=True)\n"
                    " time.sleep(30)\n"
                    "if 'list' in sys.argv[1:3]:\n"
                    " print(json.dumps({'rust-suites': {'p::smoke': {'testcases': {"
                    "'fast_case': {'ignored': False, 'filter-match': {'status': 'matches'}},"
                    "'slow_case': {'ignored': False, 'filter-match': {'status': 'mismatch'}}}}}}))\n"
                )
                cargo.chmod(0o755)
                with (
                    mock.patch.dict(os.environ, {
                        "PATH": str(root) + os.pathsep + os.environ["PATH"],
                        "CARGO_TARGET_DIR": str(root / "target"),
                    }),
                    mock.patch.object(owned, "_commit", return_value="b" * 40),
                    mock.patch.object(owned, "EXPANSION_EXECUTION_SECONDS", 0.5),
                ):
                    receipt = owned.execute_shard(
                        plan, lane="package-expansion",
                        shard=owned.shard_for(owned.Identity("p", "smoke")),
                        output=root / "receipt", repo=root,
                    )
                self.assertFalse(receipt["success"])
                self.assertIn("execution deadline", " ".join(receipt["failures"]))
                self.assertEqual(owned.load_receipts(root / "receipt"), [receipt])
                commands = json.loads((root / "receipt/commands.json").read_text())
                self.assertEqual(commands[-1]["returncode"], 124)
                self.assertEqual(commands[-1]["stdout"], "deadline output\n")
                self.assertEqual(receipt["executed_tests"], [])

    def test_success_and_nonzero_exit_preserve_command_output(self) -> None:
        for code in (0, 7):
            command = [sys.executable, "-c", f"import sys; print('out'); print('err', file=sys.stderr); sys.exit({code})"]
            kwargs = dict(cwd=Path.cwd(), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if code:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    owned.run_command(command, **kwargs)
                result = caught.exception
            else:
                result = owned.run_command(command, **kwargs)
            self.assertEqual(result.returncode, code)
            self.assertEqual(result.stdout, "out\n")
            self.assertEqual(result.stderr, "err\n")

    def test_timeout_terminates_a_child_that_inherits_output_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child-survived"
            child = f"import pathlib,time; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('alive')"
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                "print('started', flush=True); print('stderr', file=sys.stderr, flush=True); time.sleep(30)"
            )
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                owned.run_command(
                    [sys.executable, "-c", parent], cwd=Path(tmp), check=True,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=0.5,
                )
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("started", caught.exception.stdout)
            self.assertIn("stderr", caught.exception.stderr)
            time.sleep(1.6)
            self.assertFalse(marker.exists(), "the timed-out command left its child alive")


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "version": 1,
            "mode": "push",
            "base_sha": "a" * 40,
            "candidate_sha": "b" * 40,
            "event_pr_head": None,
            "changed_records": [],
            "path_dispositions": [],
            "target_dispositions": [],
            "selected_packages": ["p", "q"],
            "eligible_targets": ["p::smoke", "q::smoke"],
            "change_owned": ["p::smoke", "q::smoke"],
            "package_expansion": [],
            "standing_targets": [],
            "target_exclusions": [{"identity": "p::heavy", "owner": OWNER}],
            "test_exclusions": [
                {"identity": "p::smoke::slow_case", "owner": OWNER}
            ],
            "shards": {
                "change_owned": owned.shard_map(
                    [owned.Identity("p", "smoke"), owned.Identity("q", "smoke")]
                ),
                "package_expansion": owned.shard_map([]),
            },
        }
        owned.attach_plan_digest(self.plan)

    def receipts(self, *, surface: str = "change_owned") -> list[dict]:
        result = []
        for shard in range(4):
            selected = self.plan["shards"][surface][str(shard)]
            tests = [f"{identity}::fast_case" for identity in selected]
            receipt = {
                "version": 1,
                "lane": surface.replace("_", "-"),
                "shard": shard,
                "plan_digest": self.plan["plan_digest"],
                "selected_targets": selected,
                "executed_targets": selected,
                "selected_tests": tests,
                "executed_tests": tests,
                "commands_file": "commands.json",
                "timings_file": "timings.json",
                "test_list_file": "test-list.json",
                "junit_file": "junit.xml",
                "started_at": "2026-09-13T00:00:00Z",
                "finished_at": "2026-09-13T00:00:01Z",
                "elapsed_seconds": 1.0,
                "soft_budget_seconds": (
                    900 if surface == "package_expansion" else None
                ),
                "soft_budget_exceeded": False,
                "sidecars": {
                    "commands.json": "0" * 64,
                    "timings.json": "1" * 64,
                    "test-list.json": "2" * 64,
                    "junit.xml": "3" * 64,
                },
                "success": True,
                "failures": [],
            }
            owned.attach_receipt_digest(receipt)
            result.append(receipt)
        return result

    def assert_required_fails(self, mutate) -> None:
        receipts = self.receipts()
        mutate(receipts)
        with self.assertRaises(ValueError):
            owned.validate_change_owned_report(self.plan, receipts)

    def test_required_report_accepts_exact_four_shard_cover(self) -> None:
        report = owned.validate_change_owned_report(self.plan, self.receipts())
        self.assertTrue(report["success"])
        self.assertEqual(report["covered_targets"], sorted(self.plan["change_owned"]))

    def test_required_report_rejects_missing_digest_duplicate_uncovered_excluded_and_failure(self) -> None:
        mutations = [
            lambda rows: rows.pop(),
            lambda rows: rows[0].update(plan_digest="bad"),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(executed_targets=[]),
            lambda rows: rows[0]["executed_targets"].append("p::heavy"),
            lambda rows: rows[0].update(success=False, failures=["command failed"]),
            lambda rows: rows[0]["executed_tests"].append("p::smoke::slow_case"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_required_fails(mutation)

    def test_receipt_digest_mismatch_is_rejected(self) -> None:
        self.assert_required_fails(
            lambda rows: rows[0]["selected_targets"].append("p::invented")
        )

    def test_informational_summary_records_failures_but_does_not_raise(self) -> None:
        self.plan["package_expansion"] = ["p::default_gated"]
        self.plan["eligible_targets"].append("p::default_gated")
        self.plan["shards"]["package_expansion"] = owned.shard_map(
            [owned.Identity("p", "default_gated")]
        )
        owned.attach_plan_digest(self.plan)
        receipts = self.receipts(surface="package_expansion")
        receipts[0]["success"] = False
        receipts[0]["failures"] = ["timeout"]
        owned.attach_receipt_digest(receipts[0])
        summary = owned.summarize_package_expansion(self.plan, receipts[:-1])
        self.assertFalse(summary["observed_success"])
        self.assertFalse(summary["required"])
        self.assertTrue(summary["failures"])

    def test_informational_summary_records_soft_budget_overrun(self) -> None:
        self.plan["package_expansion"] = ["p::default_gated"]
        self.plan["eligible_targets"].append("p::default_gated")
        self.plan["shards"]["package_expansion"] = owned.shard_map(
            [owned.Identity("p", "default_gated")]
        )
        owned.attach_plan_digest(self.plan)
        receipts = self.receipts(surface="package_expansion")
        receipts[0]["elapsed_seconds"] = 901.0
        receipts[0]["soft_budget_exceeded"] = True
        owned.attach_receipt_digest(receipts[0])
        summary = owned.summarize_package_expansion(self.plan, receipts)
        self.assertFalse(summary["observed_success"])
        self.assertTrue(
            any("soft budget" in finding for finding in summary["failures"])
        )

    def test_load_receipts_rejects_missing_or_modified_sidecars(self) -> None:
        receipt = self.receipts()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard"
            shard.mkdir()
            for index, name in enumerate(
                ("commands.json", "timings.json", "test-list.json", "junit.xml")
            ):
                content = f"sidecar-{index}".encode()
                (shard / name).write_bytes(content)
                receipt["sidecars"][name] = owned.sha256_bytes(content)
            owned.attach_receipt_digest(receipt)
            (shard / "receipt.json").write_bytes(owned.canonical_json(receipt))
            self.assertEqual(len(owned.load_receipts(root)), 1)
            (shard / "commands.json").write_text("modified")
            with self.assertRaisesRegex(ValueError, "sidecar digest"):
                owned.load_receipts(root)

    def test_informational_cli_records_malformed_receipts_and_returns_zero(self) -> None:
        self.plan["package_expansion"] = ["p::default_gated"]
        self.plan["eligible_targets"].append("p::default_gated")
        self.plan["shards"]["package_expansion"] = owned.shard_map(
            [owned.Identity("p", "default_gated")]
        )
        owned.attach_plan_digest(self.plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "plan.json"
            receipt_root = root / "receipts"
            output = root / "report"
            receipt_root.mkdir()
            plan_path.write_bytes(owned.canonical_json(self.plan))
            (receipt_root / "receipt.json").write_text("{malformed")
            result = owned.main(
                [
                    "report",
                    "--plan",
                    str(plan_path),
                    "--lane",
                    "package-expansion",
                    "--receipts-root",
                    str(receipt_root),
                    "--output",
                    str(output),
                ]
            )
            report = json.loads((output / "report.json").read_text())
        self.assertEqual(result, 0)
        self.assertFalse(report["observed_success"])
        self.assertTrue(report["failures"])


if __name__ == "__main__":
    unittest.main()
