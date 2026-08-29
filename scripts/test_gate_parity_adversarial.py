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

It also covers bare `chelis ...` invocations. RT-2 found that the
parity parser originally only inspected commands starting with
`cargo `, so a gate job that hand-inlined a bare `chelis ...` command
(rather than the `cargo run -p chelis-cli ... -- ...` form) slipped
past the lock, contradicting the "every `cargo`/`chelis` invocation"
claim in the `test_gate.py` docstring and the design note.
`_is_gate_relevant_command` in `scripts/test_gate.py` now matches both
prefixes; `test_bare_chelis_command_is_caught` is the adversarial proof
that the gap is closed.
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

    def test_cargo_inside_plain_continuation_is_caught(self):
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky plain continuation\n"
            "        run:\n"
            "          python3 scripts/gate.py integration --support-only;\n"
            "          cargo check -p chelis-types\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch cargo in a plain continued scalar",
        )

    def test_quoted_cargo_command_with_yaml_comment_is_caught(self):
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky quoted command\n"
            "        run: 'cargo check -p chelis-types' # valid YAML comment\n\n"
            + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch quoted cargo before a YAML comment",
        )

    def test_escaped_double_quoted_cargo_command_is_caught(self):
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky escaped command\n"
            '        run: "\\x63argo check -p chelis-types"\n\n'
            + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT fail closed on a YAML-escaped command",
        )

    def test_underscore_job_id_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            "\n  Unclassified_job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored a valid job id containing underscore "
            "and uppercase characters",
        )

    def test_quoted_job_id_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            '\n  "Quoted_Job":\n'
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored a valid quoted job id",
        )

    def test_anchored_job_with_direct_cargo_is_caught(self):
        mutated = self.ci_text.rstrip() + (
            "\n  Hidden_Job: &hidden_job\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock ignored an anchored job",
        )

    def test_bare_chelis_command_is_caught(self):
        # RT-2 finding, now closed: the parser originally only inspected
        # commands starting with `cargo `, so a gate job that
        # hand-inlined a bare `chelis ...` command (rather than the
        # `cargo run -p chelis-cli ... -- ...` form) slipped past the
        # lock. `_is_gate_relevant_command` now matches `chelis ` too.
        # Plant a hand-inlined bare `chelis` step into the `integration`
        # gate job. The parity lock MUST fail.
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky bare chelis call\n"
            "        run: chelis lint --check .\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch a hand-inlined bare `chelis` "
            "command in the `integration` gate job -- the RT-2 gap is "
            "still open",
        )

    def test_cargo_run_chelis_cli_command_is_caught(self):
        # The other `chelis` invocation shape: `cargo run -p chelis-cli
        # --bin chelis -- ...`. This already starts with `cargo `, so
        # the `cargo ` prefix catches it, but pin it explicitly so the
        # coverage of both `chelis` invocation forms is visible.
        mutated = self.ci_text.replace(
            ANCHOR,
            "      - name: Sneaky cargo-run chelis call\n"
            "        run: cargo run -p chelis-cli --bin chelis -- "
            "lint --check .\n\n" + ANCHOR,
            1,
        )
        self.assertNotEqual(mutated, self.ci_text, "mutation did not apply")
        result = _run_parity_against(mutated)
        self.assertGreater(
            len(result.failures) + len(result.errors),
            0,
            "the parity lock did NOT catch a hand-inlined `cargo run -p "
            "chelis-cli --bin chelis -- ...` command in the "
            "`integration` gate job",
        )


if __name__ == "__main__":
    unittest.main()
