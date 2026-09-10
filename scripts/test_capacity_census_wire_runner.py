"""C6 requires execution receipts, not successful commands or empty suites."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from capacity_census_graph import GraphError


class LibtestReceipts(unittest.TestCase):
    def test_failed_native_execution_reports_its_actual_output(self):
        from capacity_census_wire_runner import run_libtest

        events = [
            {"type": "suite", "event": "started", "test_count": 1},
            {"type": "test", "event": "started", "name": "facade"},
            {"type": "test", "event": "failed", "name": "facade",
             "stdout": "ModuleNotFoundError: No module named 'numpy'"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "facade"
            binary.write_bytes(b"fixture")
            result = subprocess.CompletedProcess(
                [str(binary)], 101,
                "\n".join(json.dumps(event) for event in events).encode(),
                b"native facade failed\n",
            )
            with patch("capacity_census_wire_runner.subprocess.run", return_value=result):
                with self.assertRaisesRegex(GraphError, "ModuleNotFoundError.*numpy") as caught:
                    run_libtest(root, binary, ("facade",))
            self.assertIn("native facade failed", str(caught.exception))

    def test_cargo_artifact_must_be_the_unique_requested_test_in_this_target(self):
        from capacity_census_wire_runner import select_test_binary

        root = Path("/workspace")
        target = root / "target/census"
        artifact = {
            "reason": "compiler-artifact",
            "target": {
                "name": "wire",
                "kind": ["test"],
                "src_path": "/workspace/crates/api/tests/wire.rs",
            },
            "profile": {"test": True},
            "executable": "/workspace/target/census/debug/deps/wire-abc",
        }
        source = root / "crates/api/tests/wire.rs"
        self.assertEqual(
            select_test_binary(target, source, "wire", [artifact]),
            Path(artifact["executable"]),
        )
        for records in (
            [],
            [artifact, artifact],
            [{**artifact, "executable": "/elsewhere/wire"}],
            [{**artifact, "profile": {"test": False}}],
            [{**artifact, "target": {**artifact["target"], "name": "other"}}],
            [
                {
                    **artifact,
                    "target": {**artifact["target"], "src_path": "/elsewhere/wire.rs"},
                }
            ],
        ):
            with self.subTest(records=records), self.assertRaises(GraphError):
                select_test_binary(target, source, "wire", records)

    def test_library_artifact_requires_exact_library_kind_and_test_profile(self):
        from capacity_census_wire_runner import select_test_binary

        target = Path("/workspace/target/census")
        source = Path("/workspace/crates/api/src/lib.rs")
        artifact = {
            "reason": "compiler-artifact",
            "target": {"name": "api", "kind": ["lib"], "src_path": str(source)},
            "profile": {"test": True},
            "executable": str(target / "debug/deps/api-abc"),
        }
        self.assertEqual(select_test_binary(target, source, "api", [artifact], kind="lib"), Path(artifact["executable"]))
        for records in ([], [artifact, artifact],
            [{**artifact, "profile": {"test": False}}],
            [{**artifact, "target": {**artifact["target"], "kind": ["test"]}}],
            [{**artifact, "target": {**artifact["target"], "kind": ["bin"]}}],
            [{**artifact, "target": {**artifact["target"], "src_path": "/foreign/lib.rs"}}],
            [{**artifact, "executable": "/foreign/api-abc"}],
        ):
            with self.subTest(records=records), self.assertRaises(GraphError):
                select_test_binary(target, source, "api", records, kind="lib")
        with self.assertRaises(GraphError):
            select_test_binary(target, source, "api", [artifact])
        with self.assertRaises(GraphError):
            select_test_binary(target, source, "api", [artifact], kind="bin")

    def test_actual_rust_framework_does_not_accept_ignored_or_zero_match(self):
        from capacity_census_wire_runner import run_libtest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = root / "controls.rs", root / "controls"
            source.write_text(
                "#[test] fn accepts() { assert_eq!(2 + 2, 4); }\n"
                '#[test] fn rejects() { assert!("bad".parse::<i32>().is_err()); }\n'
                "#[test] #[ignore] fn ignored() {}\n"
                "#[test] fn early_exit() { std::process::exit(0); }\n"
                '#[test] fn fails() { panic!("must fail"); }\n'
                "#[test] fn managed_python() {\n"
                ' assert_eq!(std::env::var("PYO3_PYTHON"), std::env::var("EXPECTED_PYTHON"));\n'
                ' assert_eq!(std::env::var("VIRTUAL_ENV"), std::env::var("EXPECTED_VENV"));\n'
                "}\n"
            )
            subprocess.run(
                ["rustc", "--test", str(source), "-o", str(binary)],
                check=True,
                capture_output=True,
            )
            receipt = run_libtest(root, binary, ("accepts", "rejects"))
            self.assertEqual(receipt.executed, ("accepts", "rejects"))
            with patch.dict(
                os.environ,
                {
                    "PYO3_PYTHON": "/unrelated/python",
                    "VIRTUAL_ENV": "/unrelated/venv",
                    "EXPECTED_PYTHON": sys.executable,
                    "EXPECTED_VENV": sys.prefix,
                },
            ):
                run_libtest(root, binary, ("managed_python",))
            for selected in (("absent",), ("ignored",), ("early_exit",), ("fails",)):
                with self.subTest(selected=selected), self.assertRaises(GraphError):
                    run_libtest(root, binary, selected)

    def test_exact_positive_selection_and_execution_are_required(self):
        from capacity_census_wire_runner import validate_libtest_execution

        events = [
            {"type": "suite", "event": "started", "test_count": 2},
            {"type": "test", "event": "started", "name": "accept"},
            {"type": "test", "event": "ok", "name": "accept"},
            {"type": "test", "event": "started", "name": "reject"},
            {"type": "test", "event": "ok", "name": "reject"},
            {
                "type": "suite",
                "event": "ok",
                "passed": 2,
                "failed": 0,
                "ignored": 0,
                "measured": 0,
                "filtered_out": 3,
            },
        ]
        self.assertEqual(
            validate_libtest_execution(("accept", "reject"), events),
            ("accept", "reject"),
        )
        for mutation in (
            events[:3] + events[-1:],
            events[:-1],
            events[:3] + [events[2]] + events[3:],
            [
                {**event, "event": "ignored"}
                if event.get("name") == "reject" and event["event"] == "ok"
                else event
                for event in events
            ],
            [
                {**event, "event": "failed"}
                if event.get("name") == "reject" and event["event"] == "ok"
                else event
                for event in events
            ],
            events[1:],
            [events[0], events[2], *events[3:]],
            [*events[:-1], {**events[-1], "ignored": 1}],
            [*events[:-1], {**events[-1], "passed": 1}],
            [{**events[0], "test_count": 0}, *events[1:]],
        ):
            with self.subTest(events=mutation), self.assertRaises(GraphError):
                validate_libtest_execution(("accept", "reject"), mutation)
        for selection in ((), ("accept", "accept"), ("absent",)):
            with self.subTest(selection=selection), self.assertRaises(GraphError):
                validate_libtest_execution(selection, events)


class SupervisedUnittest(unittest.TestCase):
    def run_fixture(self, body):
        from capacity_census_wire_runner import run_python_tests

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.py").write_text(body)
            return run_python_tests(
                root, ("fixture.Cases.test_accept", "fixture.Cases.test_reject")
            )

    def test_framework_records_both_real_test_bodies(self):
        receipt = self.run_fixture(
            "import unittest\n"
            "class Cases(unittest.TestCase):\n"
            " def test_accept(self): self.assertEqual(2 + 2, 4)\n"
            " def test_reject(self):\n"
            "  with self.assertRaises(ValueError): int('bad')\n"
        )
        self.assertEqual(receipt.selected, receipt.executed)
        self.assertEqual(len(receipt.executed), 2)
        self.assertEqual(len(receipt.output_sha256), 64)

    def test_skips_missing_tests_errors_and_early_exit_cannot_pass(self):
        base = (
            "import unittest\n"
            "class Cases(unittest.TestCase):\n"
            " def test_accept(self): self.assertTrue(True)\n"
        )
        for body in (
            base,
            base + " @unittest.skip('not run')\n def test_reject(self): pass\n",
            base + " def test_reject(self): self.fail('bad')\n",
            base + " def test_reject(self): raise RuntimeError('bad')\n",
            base
            + " @unittest.expectedFailure\n def test_reject(self): self.fail('bad')\n",
            base + " def test_reject(self):\n  import os\n  os._exit(0)\n",
            base + " def test_reject(self):\n  with self.subTest(): self.fail('bad')\n",
            base
            + " @classmethod\n def setUpClass(cls): raise unittest.SkipTest('class skip')\n"
            " def test_reject(self): pass\n",
        ):
            with self.subTest(body=body), self.assertRaises(GraphError):
                self.run_fixture(body)

    def test_duplicate_empty_and_modified_selection_are_rejected(self):
        from capacity_census_wire_runner import run_python_tests

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.py").write_text(
                "import unittest\n"
                "class Cases(unittest.TestCase):\n"
                " def test_accept(self): pass\n"
                " def test_reject(self): pass\n"
                "def load_tests(loader, suite, pattern): return unittest.TestSuite()\n"
            )
            for names in ((), ("fixture.Cases.test_accept",) * 2):
                with self.assertRaises(GraphError):
                    run_python_tests(root, names)
            # Explicit method loading bypasses load_tests hooks and still executes
            # the exact obligations; replacing a module's suite cannot erase them.
            receipt = run_python_tests(
                root, ("fixture.Cases.test_accept", "fixture.Cases.test_reject")
            )
            self.assertEqual(len(receipt.executed), 2)


if __name__ == "__main__":
    unittest.main()
