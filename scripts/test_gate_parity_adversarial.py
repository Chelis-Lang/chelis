"""Adversarial coverage for the `gate.py` CI-parity lock (PR #125).

Run via: `python3 -m unittest scripts.test_gate_parity_adversarial`
from repo root, or `python3 scripts/test_gate_parity_adversarial.py`.

`scripts/test_gate.py` ships the parity lock: it greps
`.github/workflows/ci.yml` and fails if a gate job hand-inlines a
command that `gate.py` does not produce. `scripts/test_gate.py`'s own
tests assert it passes on the *current* workflow. This file is the
adversarial complement: it MUTATES a copy of `ci.yml`, points the
parity test at the mutation, and asserts the lock actually fails. A
parity lock that never fails on a real drift is theater.

It also pins one KNOWN GAP discovered by red-team: the parity parser
only inspects commands that start with `cargo ` (see
`_parse_ci_gate_invocations` in `scripts/test_gate.py`, the
`command.startswith("cargo ")` filter). The module docstring and the
design note both describe the lock as covering "every `cargo`/`chelis`
invocation", but a gate job that hand-inlines a bare `chelis ...`
command (rather than `cargo run -p chelis-cli ... -- ...`) slips past
the lock. `test_known_gap_bare_chelis_command_is_not_caught` documents
that gap as an executable expectation: if a future fix closes it, this
test flips to a failure and forces the docstring/design note to be
reconciled with the parser.
"""

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ANCHOR = "      - name: Test-timing budget (informational)"


def _load_test_gate():
    """Load `scripts/test_gate.py` as a module so we can drive its
    `CiParityTests` against a mutated workflow file."""
    spec = importlib.util.spec_from_file_location(
        "test_gate", REPO_ROOT / "scripts" / "test_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["test_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_parity_against(workflow_text: str) -> unittest.TestResult:
    """Write `workflow_text` to a temp file, point `test_gate.CI_YML` at
    it, and run the whole `CiParityTests` suite. Returns the result."""
    tg = _load_test_gate()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ci.yml"
        path.write_text(workflow_text)
        tg.CI_YML = path
        suite = unittest.TestLoader().loadTestsFromTestCase(tg.CiParityTests)
        return unittest.TextTestRunner(
            stream=io.StringIO(), verbosity=0
        ).run(suite)


class GateParityAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")
        self.ci_text = CI_YML.read_text()
        self.assertIn(
            ANCHOR,
            self.ci_text,
            "ci.yml anchor moved; update ANCHOR in this adversarial test",
        )

    def test_control_unmutated_workflow_passes_parity(self):
        # Sanity floor: the real workflow must pass the lock, otherwise
        # the mutation tests below prove nothing.
        result = _run_parity_against(self.ci_text)
        self.assertEqual(
            (len(result.failures), len(result.errors)),
            (0, 0),
            "the unmutated ci.yml fails the parity lock; the lock is "
            "already broken independent of any mutation",
        )

    def test_hand_inlined_cargo_command_is_caught(self):
        # Plant a hand-inlined `cargo test` step into the `integration`
        # gate job. The parity lock MUST fail.
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky hand-inlined cargo step\n"
            "        run: cargo test --workspace --doc\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch a hand-inlined `cargo test` "
            "command in the `integration` gate job -- the lock is "
            "theater",
        )

    def test_cargo_inside_multiline_run_block_is_caught(self):
        # A `run: |` multiline block would hide its commands from the
        # line-based parser; `test_no_multiline_run_in_gate_jobs` is the
        # backstop. Confirm it actually fires.
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky multiline\n"
            "        run: |\n"
            "          cargo test --workspace --doc\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch a `cargo` command hidden in "
            "a `run: |` multiline block in a gate job",
        )

    def test_known_gap_bare_chelis_command_is_not_caught(self):
        # KNOWN GAP (red-team finding): the parser only inspects
        # commands starting with `cargo `. A gate job that hand-inlines
        # a bare `chelis ...` command slips past the lock, even though
        # the design note and `test_gate.py`'s docstring both say the
        # lock covers "every `cargo`/`chelis` invocation".
        #
        # This test pins the gap as an executable expectation: it
        # asserts the lock currently does NOT catch the bare-`chelis`
        # mutation. If a future change extends the parser to also match
        # `chelis ` (closing the gap), this test FAILS -- which is the
        # signal to delete it and reconcile the docs.
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky bare chelis call\n"
            "        run: chelis lint --check .\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertEqual(
            (len(result.failures), len(result.errors)),
            (0, 0),
            "the parity lock now CATCHES a bare `chelis` command -- the "
            "known gap is closed. Delete this test and update the "
            "`test_gate.py` docstring + design note, which already "
            "claim `cargo`/`chelis` coverage",
        )


if __name__ == "__main__":
    unittest.main()
