from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("hash_order_phase_a_oracle.py")
SPEC = importlib.util.spec_from_file_location("hash_order_phase_a_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class HashOrderPhaseAOracleTests(unittest.TestCase):
    def test_component_list_is_exact_and_includes_executable_parity(self) -> None:
        self.assertEqual(
            ORACLE.COMMANDS,
            (
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-compiler-api",
                    "cache_format_version_tracks_the_deferred_ledger_removal",
                    "--no-fail-fast",
                ),
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-cli",
                    "--test",
                    "cli",
                    "executable_examples",
                    "--no-fail-fast",
                ),
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-types",
                    "--test",
                    "issue_942_inferred_tensor_cast",
                    "--no-fail-fast",
                ),
                (
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-cli",
                    "--test",
                    "parity",
                    "parity_hash_order_determinism",
                    "--no-fail-fast",
                ),
            ),
        )

    def test_runs_every_component_and_prints_one_pass_marker(self) -> None:
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate(runner=runner)
        self.assertEqual(seen, list(ORACLE.COMMANDS))
        self.assertEqual(output.getvalue(), "HASH ORDER PHASE A ORACLE: PASS\n")

    def test_stops_at_the_first_failed_component(self) -> None:
        commands = (("first",), ("second",), ("third",))
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 7 if command == ("second",) else 0)

        with self.assertRaisesRegex(ORACLE.PhaseAOracleFailure, "exited 7: second"):
            ORACLE.validate(runner=runner, commands=commands)
        self.assertEqual(seen, [("first",), ("second",)])


if __name__ == "__main__":
    unittest.main()
