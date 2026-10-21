from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


spec = importlib.util.spec_from_file_location(
    "no_std_core_support", Path(__file__).with_name("no_std_core_support.py")
)
assert spec is not None and spec.loader is not None
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)


def metadata(*, missing: bool = False) -> dict:
    packages = [
        {"id": name, "name": name, "targets": [{"kind": ["lib"]}]}
        for name in ("consumer", "core", "leaf", "host-build", "dev-only")
    ]
    nodes = [
        {
            "id": "consumer",
            "deps": [
                {"pkg": "core", "dep_kinds": [{"kind": None}]},
                {"pkg": "host-build", "dep_kinds": [{"kind": "build"}]},
                {"pkg": "dev-only", "dep_kinds": [{"kind": "dev"}]},
            ],
        },
        {"id": "core", "deps": [{"pkg": "leaf", "dep_kinds": [{"kind": None}]}]},
        {"id": "leaf", "deps": []},
        {"id": "host-build", "deps": []},
        {"id": "dev-only", "deps": []},
    ]
    for node in nodes:
        node["features"] = []
    if missing:
        nodes = [node for node in nodes if node["id"] != "leaf"]
    return {"packages": packages, "resolve": {"root": "consumer", "nodes": nodes}}


def test_inventory() -> dict:
    return {
        "rust-suites": {
            "core": {
                "package-name": "core",
                "binary-name": "core",
                "testcases": {
                    "tests::mandatory": {
                        "kind": "test",
                        "ignored": False,
                        "filter-match": {"status": "matches"},
                    }
                },
            }
        }
    }


def compiler_error(target: str, source: str) -> str:
    return json.dumps({
        "reason": "compiler-message",
        "target": {"src_path": target},
        "message": {
            "level": "error",
            "message": "unresolved import `retired::Thing`",
            "code": {"code": "E0432"},
            "spans": [{
                "file_name": "src/lib.rs",
                "line_start": 1,
                "is_primary": True,
                "text": [{"text": source}],
                "label": "no Thing in the root",
            }],
            "children": [],
        },
    })


class OracleFailureTests(unittest.TestCase):
    def test_failed_real_command_fails_acceptance(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "exit 23"):
            support.run((sys.executable, "-c", "raise SystemExit(23)"))

    def test_absent_executable_fails_acceptance(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "cannot execute"):
            support.run(("/chelis-oracle-absent-executable",))

    def test_normal_dependency_closure_excludes_host_tools(self) -> None:
        self.assertEqual(
            support._normal_packages(metadata()),
            {"consumer", "core", "leaf"},
        )

    def test_reachable_std_feature_fails_portable_acceptance(self) -> None:
        graph = metadata()
        graph["resolve"]["nodes"][2]["features"] = ["std"]
        with self.assertRaisesRegex(support.OracleFailure, "target std feature: leaf"):
            support._normal_packages(graph)

    def test_host_tool_features_do_not_taint_the_target_graph(self) -> None:
        graph = metadata()
        graph["resolve"]["nodes"][3]["features"] = ["std"]
        graph["resolve"]["nodes"][4]["features"] = ["std"]
        self.assertEqual(
            support._normal_packages(graph),
            {"consumer", "core", "leaf"},
        )

    def test_incomplete_dependency_graph_fails_closed(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "incomplete dependency graph"):
            support._normal_packages(metadata(missing=True))

    def test_absent_resolution_fails_closed(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "incomplete dependency graph"):
            support._normal_packages({"packages": [], "resolve": None})

    def test_inventory_preserves_the_binary_and_case_identity(self) -> None:
        self.assertEqual(
            support._selected_test_names(test_inventory()),
            {"core::core$tests::mandatory"},
        )

    def test_ignored_mandatory_case_fails_acceptance(self) -> None:
        inventory = test_inventory()
        inventory["rust-suites"]["core"]["testcases"]["tests::mandatory"]["ignored"] = True
        with self.assertRaisesRegex(support.OracleFailure, "ignored mandatory test"):
            support._selected_test_names(inventory)

    def test_removed_named_case_fails_with_other_cases_present(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "omitted or filtered"):
            support._selected_test_names(test_inventory(), ("tests::removed",))

    def test_missing_result_cannot_hide_behind_another_pass(self) -> None:
        selected = {"core::core$tests::mandatory", "core::core$tests::other"}
        result = "\n".join(json.dumps(record) for record in (
            {"type": "test", "event": "ignored", "name": "core::core$manual"},
            {"type": "test", "event": "ok", "name": "core::core$tests::other"},
        ))
        with self.assertRaisesRegex(support.OracleFailure, "missing"):
            support._check_test_results(result, selected)

    def test_ignored_result_cannot_establish_execution(self) -> None:
        result = json.dumps({
            "type": "test", "event": "ignored", "name": "core::core$tests::mandatory",
        })
        with self.assertRaisesRegex(support.OracleFailure, "did not pass"):
            support._check_test_results(result, {"core::core$tests::mandatory"})

    def test_dependency_error_cannot_satisfy_a_negative_fixture(self) -> None:
        source = "use retired::Thing;"
        diagnostic = compiler_error("/dependency/src/lib.rs", source)
        with self.assertRaisesRegex(support.OracleFailure, "no fixture-local error"):
            support._fixture_diagnostics(diagnostic, Path("/fixture"), source)

    def test_wrong_source_span_cannot_satisfy_a_negative_fixture(self) -> None:
        diagnostic = compiler_error("/fixture/src/lib.rs", "use unrelated::Thing;")
        with self.assertRaisesRegex(support.OracleFailure, "no fixture-local error"):
            support._fixture_diagnostics(diagnostic, Path("/fixture"), "use retired::Thing;")

    def test_invalid_compiler_json_fails_acceptance(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "invalid negative compile evidence"):
            support._fixture_diagnostics("not JSON", Path("/fixture"), "use retired::Thing;")


if __name__ == "__main__":
    unittest.main()
