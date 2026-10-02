"""The ownership-ledger commands derive their targets from Cargo metadata."""

from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path
import tomllib
import unittest
from unittest import mock

from scripts import gate
from scripts import ownership_ledger_tests as ledger


ROOT = Path(__file__).resolve().parents[1]


def metadata(*packages: tuple[str, list[dict]], members: int | None = None) -> dict:
    rows = [
        {"id": f"{name} 0.1.0", "name": name, "targets": targets}
        for name, targets in packages
    ]
    return {
        "packages": rows,
        "workspace_members": [row["id"] for row in rows[:members]],
    }


def test_target(name: str, features: list[str] | None = None, kind=("test",)) -> dict:
    row = {"name": name, "kind": list(kind)}
    if features is not None:
        row["required-features"] = features
    return row


LEDGER = ["ownership-ledger"]
BASE = (
    (
        "chelis-compiler-api",
        [
            test_target("key_root_lanes", LEDGER),
            test_target("builtin_named_kernel_inputs", LEDGER),
            test_target("featureless"),
            test_target("emission_observer", ["compilation-trace"]),
        ],
    ),
    ("chelis-cli", [test_target("issue_1314_json_bigint_ledger", LEDGER)]),
    ("chelis-runtime", [test_target("runtime_unit")]),
)


def declared_ledger_targets() -> set[tuple[str, str]]:
    """Read every member manifest directly, independently of Cargo."""
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    declared = set()
    for member in workspace["members"]:
        manifest = tomllib.loads((ROOT / member / "Cargo.toml").read_text())
        for target in manifest.get("test", []):
            if ledger.LEDGER_FEATURE in target.get("required-features", []):
                declared.add((manifest["package"]["name"], target["name"]))
    return declared


class DerivationTests(unittest.TestCase):
    def test_commands_list_each_package_ledger_targets_in_sorted_order(self) -> None:
        source = metadata(*BASE)

        self.assertEqual(
            ledger.ledger_command("chelis-compiler-api", source),
            [
                "cargo", "nextest", "run", "-p", "chelis-compiler-api",
                "--features", "ownership-ledger",
                "--test", "builtin_named_kernel_inputs",
                "--test", "key_root_lanes",
            ],
        )
        self.assertEqual(
            ledger.ledger_command("chelis-cli", source),
            [
                "cargo", "nextest", "run", "-p", "chelis-cli",
                "--features", "ownership-ledger",
                "--test", "issue_1314_json_bigint_ledger",
            ],
        )

    def test_a_new_ledger_target_joins_its_command_without_a_list_edit(self) -> None:
        packages = [list(row) for row in BASE]
        packages[0][1] = [*packages[0][1], test_target("added_ledger_case", LEDGER)]

        command = ledger.ledger_command(
            "chelis-compiler-api", metadata(*map(tuple, packages))
        )

        self.assertIn("added_ledger_case", command)

    def test_targets_the_command_cannot_run_are_rejected(self) -> None:
        cases = {
            "extra feature": (
                "chelis-cli",
                test_target("both", ["ownership-ledger", "other"]),
                "activates only",
            ),
            "not an integration test": (
                "chelis-cli",
                test_target("bench", LEDGER, kind=("bench",)),
                "not an integration test",
            ),
            "unplaced package": (
                "chelis-runtime",
                test_target("runtime_ledger", LEDGER),
                "no ledger command runs",
            ),
        }
        for label, (package, target, message) in cases.items():
            with self.subTest(case=label):
                packages = [
                    (name, [*targets, target] if name == package else targets)
                    for name, targets in BASE
                ]
                with self.assertRaisesRegex(ledger.LedgerTargetError, message):
                    ledger.ledger_targets(metadata(*packages))

    def test_a_ledger_package_without_targets_is_rejected(self) -> None:
        # An empty --test list would run the package's whole test suite.
        packages = [
            (name, [] if name == "chelis-cli" else targets)
            for name, targets in BASE
        ]
        with self.assertRaisesRegex(ledger.LedgerTargetError, "no ownership-ledger"):
            ledger.ledger_targets(metadata(*packages))
        with self.assertRaisesRegex(ledger.LedgerTargetError, "not a ledger package"):
            ledger.ledger_command("chelis-runtime", metadata(*BASE))

    def test_packages_outside_the_workspace_are_ignored(self) -> None:
        packages = [*BASE, ("vendored", [test_target("vendored_ledger", LEDGER)])]

        self.assertEqual(
            set(ledger.ledger_targets(metadata(*packages, members=len(BASE)))),
            set(ledger.LEDGER_PACKAGES),
        )

    def test_main_prints_then_runs_the_derived_command(self) -> None:
        source = metadata(*BASE)
        expected = ledger.ledger_command("chelis-cli", source)
        for print_only in (True, False):
            with self.subTest(print_only=print_only):
                stdout = io.StringIO()
                arguments = ["chelis-cli", *(["--print"] if print_only else [])]
                with (
                    mock.patch.object(ledger, "cargo_metadata", return_value=source),
                    mock.patch.object(ledger.os, "chdir") as chdir,
                    mock.patch.object(ledger.os, "execvp") as execvp,
                    contextlib.redirect_stdout(stdout),
                ):
                    ledger.main(arguments)
                self.assertEqual(stdout.getvalue(), " ".join(expected) + "\n")
                if print_only:
                    execvp.assert_not_called()
                else:
                    chdir.assert_called_once_with(ledger.REPO_ROOT)
                    execvp.assert_called_once_with("cargo", expected)

    def test_runs_on_the_system_python_a_runner_ships(self) -> None:
        # The macOS nightly job runs this script under the runner's python3.
        source = (ROOT / "scripts/ownership_ledger_tests.py").read_text()
        tree = ast.parse(source, feature_version=(3, 9))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertLessEqual(
            imported,
            {"__future__", "argparse", "json", "os", "pathlib", "shlex",
             "subprocess", "sys", "typing"},
        )


class WorkspaceReconciliationTests(unittest.TestCase):
    def test_cargo_metadata_and_the_manifests_name_the_same_targets(self) -> None:
        derived = ledger.ledger_targets(ledger.cargo_metadata())

        self.assertEqual(
            {(package, name) for package, names in derived.items() for name in names},
            declared_ledger_targets(),
        )
        self.assertEqual(tuple(derived), ledger.LEDGER_PACKAGES)

    def test_gate_runs_every_ledger_package_in_order(self) -> None:
        commands = (gate.OWNERSHIP_LEDGER_API_TESTS, gate.OWNERSHIP_LEDGER_CLI_TESTS)
        for command in commands:
            self.assertEqual(
                command[:2], [gate.MANAGED_PYTHON, "scripts/ownership_ledger_tests.py"]
            )
        self.assertEqual(
            tuple(command[2] for command in commands), ledger.LEDGER_PACKAGES
        )
        integration = gate.STAGES["integration"]
        self.assertLess(
            integration.index(gate.OWNERSHIP_LEDGER_API_TESTS),
            integration.index(gate.OWNERSHIP_LEDGER_CLI_TESTS),
        )


if __name__ == "__main__":
    unittest.main()
