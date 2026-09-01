from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).with_name("hash_order_determinism_oracle.py")
SPEC = importlib.util.spec_from_file_location("hash_order_determinism_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class CommandRegistryTests(unittest.TestCase):
    """The registry is the contract: a leg nobody runs proves nothing."""

    def test_registers_the_configuration_closure_check(self) -> None:
        commands = [tuple(command) for command in ORACLE.COMMANDS]
        closure = [
            command
            for command in commands
            if command[-1] == "scripts/check_configuration_closure.py"
        ]
        self.assertEqual(len(closure), 1)
        # It must run first: without proven coverage of the compiled
        # configuration space, the remaining evidence is about an unbounded
        # surface.
        self.assertEqual(commands[0], closure[0])

    def test_registers_the_production_source_byte_mutation(self) -> None:
        # `cargo test --exact`, not `cargo nextest`: the repository's default
        # nextest filter can select zero tests and still exit 0.
        command = (
            "cargo",
            "test",
            "-p",
            "chelis-cli",
            "--test",
            "stdlib_typecheck_cache_oracle",
            "stale_stdlib_byte_mutation_misses_not_stale_hit",
            "--",
            "--exact",
            "--nocapture",
        )
        self.assertEqual(ORACLE.COMMANDS.count(command), 1)

    def test_registers_the_phase_b_compile_fail_controls(self) -> None:
        self.assertTrue(
            any(
                command[-1] == "scripts/check_hash_order_phase_b_compile_fail.py"
                for command in ORACLE.COMMANDS
            )
        )

    def test_every_registered_script_exists(self) -> None:
        for command in ORACLE.COMMANDS:
            for argument in command:
                if argument.endswith(".py"):
                    self.assertTrue(
                        (ORACLE.REPO_ROOT / argument).is_file(),
                        f"registered script is missing: {argument}",
                    )


class RunnerTests(unittest.TestCase):
    def test_runs_every_component_and_prints_one_pass_marker(self) -> None:
        seen: list[tuple[str, ...]] = []

        def runner(
            command: tuple[str, ...], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate(runner=runner)
        self.assertEqual(seen, [tuple(command) for command in ORACLE.COMMANDS])
        self.assertEqual(output.getvalue(), "HASH ORDER DETERMINISM ORACLE: PASS\n")

    def test_stops_at_the_first_failed_component(self) -> None:
        commands = (("first",), ("second",), ("third",))
        seen: list[tuple[str, ...]] = []

        def runner(
            command: tuple[str, ...], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(
                command, 7 if command == ("second",) else 0
            )

        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "exited 7: second"
        ):
            ORACLE.validate(commands=commands, runner=runner)
        self.assertEqual(seen, [("first",), ("second",)])

    def test_rejects_arguments(self) -> None:
        # `--scan-only` is gone with the source census it drove; a caller that
        # still passes it must fail loudly rather than run a different check.
        argv = sys.argv
        sys.argv = ["hash_order_determinism_oracle.py", "--scan-only"]
        try:
            self.assertEqual(ORACLE.main(), 2)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
