from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import no_std_core_support as support


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


class ToolEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.libdir = Path(temporary.name) / "lib/rustlib/test-host/lib"
        self.libdir.mkdir(parents=True)
        self.bindir = self.libdir.parent / "bin"
        self.bindir.mkdir()
        self.archiver = self.bindir / ("llvm-ar.exe" if os.name == "nt" else "llvm-ar")
        self.archiver.touch()
        self.archiver.chmod(0o755)
        self.environment = {"PATH": "/existing/tools"}
        environment_patch = patch.object(support, "oracle_environment", return_value=self.environment)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        run_patch = patch.object(support.subprocess, "run", return_value=subprocess.CompletedProcess(
            ["rustc", "--print", "target-libdir"], 0, str(self.libdir) + "\n", "",
        ))
        self.run = run_patch.start()
        self.addCleanup(run_patch.stop)

    def test_pinned_toolchain_installs_llvm_tools(self) -> None:
        with (support.ROOT / "rust-toolchain.toml").open("rb") as source:
            self.assertIn("llvm-tools", tomllib.load(source)["toolchain"]["components"])

    def test_matching_host_tools_precede_ambient_path(self) -> None:
        environment = support._environment()
        self.assertEqual(environment["PATH"], str(self.bindir) + os.pathsep + "/existing/tools")
        with (support.ROOT / "rust-toolchain.toml").open("rb") as source:
            channel = tomllib.load(source)["toolchain"]["channel"]
        self.assertEqual(environment["RUSTUP_TOOLCHAIN"], channel)
        self.run.assert_called_once()
        arguments, keywords = self.run.call_args
        self.assertEqual(arguments[0], ("rustc", "--print", "target-libdir"))
        self.assertEqual(keywords["env"]["RUSTUP_TOOLCHAIN"], channel)
        self.assertEqual(keywords["cwd"], support.ROOT)

    def test_explicit_compiler_and_toolchain_inputs_are_preserved(self) -> None:
        self.environment.update(RUSTC="/selected/bin/rustc", RUSTUP_TOOLCHAIN="selected",
                                PYO3_PYTHON="/selected/python")
        with patch.dict(os.environ, {"CARGO_TARGET_DIR": "/selected/target"}):
            environment = support._environment()
        self.assertEqual(environment["RUSTUP_TOOLCHAIN"], "selected")
        self.assertEqual(environment["PYO3_PYTHON"], "/selected/python")
        self.assertEqual(environment["CARGO_TARGET_DIR"], "/selected/target")
        self.assertEqual(self.run.call_args.args[0][0], "/selected/bin/rustc")
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(self.bindir))

    def test_missing_matching_archiver_rejects_even_with_ambient_tool(self) -> None:
        self.archiver.unlink()
        self.environment["PATH"] = str(self.bindir.parent)
        ambient = self.bindir.parent / self.archiver.name
        ambient.touch()
        ambient.chmod(0o755)
        with self.assertRaisesRegex(support.OracleFailure, "llvm-tools.*llvm-ar"):
            support._environment()

    @unittest.skipIf(os.name == "nt", "Windows executable permission uses file extension")
    def test_nonexecutable_matching_archiver_rejects(self) -> None:
        self.archiver.chmod(0o644)
        with self.assertRaisesRegex(support.OracleFailure, "llvm-tools.*llvm-ar"):
            support._environment()

    def test_directory_named_as_archiver_rejects(self) -> None:
        self.archiver.unlink()
        self.archiver.mkdir()
        with self.assertRaisesRegex(support.OracleFailure, "llvm-tools.*llvm-ar"):
            support._environment()

    def test_failed_compiler_discovery_rejects(self) -> None:
        self.run.return_value = subprocess.CompletedProcess([], 1, "", "toolchain unavailable")
        with self.assertRaisesRegex(support.OracleFailure, "LLVM tools.*toolchain unavailable"):
            support._environment()

    def test_missing_compiler_rejects(self) -> None:
        self.run.side_effect = FileNotFoundError("compiler unavailable")
        with self.assertRaisesRegex(support.OracleFailure, "LLVM tools.*compiler unavailable"):
            support._environment()

    def test_invalid_compiler_library_path_rejects(self) -> None:
        for stdout in ("", "relative/lib\n", str(self.libdir) + "\nextra\n",
                       str(self.libdir / "missing/lib") + "\n"):
            with self.subTest(stdout=stdout):
                self.run.return_value = subprocess.CompletedProcess([], 0, stdout, "")
                with self.assertRaisesRegex(support.OracleFailure, "invalid.*target-libdir"):
                    support._environment()


class OracleFailureTests(unittest.TestCase):
    def test_failed_real_command_fails_acceptance(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "exit 23"):
            support.run((sys.executable, "-c", "raise SystemExit(23)"), env=os.environ)

    def test_absent_executable_fails_acceptance(self) -> None:
        with self.assertRaisesRegex(support.OracleFailure, "cannot execute"):
            support.run(("/chelis-oracle-absent-executable",), env=os.environ)

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
