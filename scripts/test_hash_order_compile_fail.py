from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).with_name("check_hash_order_compile_fail.py")
SPEC = importlib.util.spec_from_file_location("check_hash_order_compile_fail", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class HashOrderCompileFailTests(unittest.TestCase):
    def test_accepts_the_required_compile_error(self) -> None:
        diagnostic = "\n".join(CHECKER.REQUIRED_DIAGNOSTICS)

        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 101, "", diagnostic)

        CHECKER.validate(runner=runner, environment={})

    def test_rejects_a_fixture_that_compiles(self) -> None:
        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, "", "")

        with self.assertRaisesRegex(
            CHECKER.HashOrderCompileFailure, "raw deferred-store iteration compiled"
        ):
            CHECKER.validate(runner=runner, environment={})

    def test_rejects_an_incomplete_diagnostic(self) -> None:
        def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 101, "", "no method named `keys`")

        with self.assertRaisesRegex(
            CHECKER.HashOrderCompileFailure, "lacks the required text"
        ):
            CHECKER.validate(runner=runner, environment={})


if __name__ == "__main__":
    unittest.main()
