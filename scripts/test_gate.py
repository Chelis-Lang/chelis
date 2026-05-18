"""Unit tests for `gate.py`.

Run via: `python3 -m unittest scripts.test_gate` from repo root,
or `python3 scripts/test_gate.py`.

Four things are locked here:

  (a) the per-stage subsets union exactly to the full canonical list;
  (b) a parity assertion: every `cargo`/`chelis` invocation in a gate
      step of `.github/workflows/ci.yml` is produced by `gate.py`. This
      covers both `cargo ...` and bare `chelis ...` commands (the
      `cargo run -p chelis-cli --bin chelis -- ...` form is caught by
      the `cargo ` prefix). This is the lock that turns future
      CI-vs-gate drift into a test failure. The non-gate jobs
      (sanitizer, macOS-smoke, docs, LOC-report, no-AI-authorship) are
      excluded by name so the exclusion is explicit and reviewable;
  (c) `--list` prints the canonical full list;
  (d) no-ai-authorship patterns cover the current banned tool identities.
"""

import importlib.util
import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gate", here / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()
REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# CI jobs that are deliberately NOT part of the per-PR developer gate.
# `gate.py` only owns the `lint-and-unit` and `integration` jobs; these
# are listed by name so the parity test's exclusion is visible.
NON_GATE_JOBS = {
    "loc-report",
    "macos-smoke",
    "backend-sanitizers",
    "no-ai-authorship",
    "docs",
}


class StageUnionTests(unittest.TestCase):
    def test_stage_subsets_union_to_full_list(self):
        union = []
        for stage in gate.STAGE_ORDER:
            union.extend(gate.STAGES[stage])
        self.assertEqual(
            union,
            gate.full_command_list(),
            "the per-stage subsets must union exactly to the full list",
        )

    def test_stage_order_covers_every_stage(self):
        self.assertEqual(
            set(gate.STAGE_ORDER),
            set(gate.STAGES.keys()),
            "STAGE_ORDER must list every stage in STAGES exactly once",
        )
        self.assertEqual(
            len(gate.STAGE_ORDER),
            len(set(gate.STAGE_ORDER)),
            "STAGE_ORDER must not repeat a stage",
        )

    def test_full_list_has_no_duplicate_commands(self):
        rendered = [gate.render(c) for c in gate.full_command_list()]
        self.assertEqual(
            len(rendered),
            len(set(rendered)),
            "no gate command should appear in more than one stage",
        )


class ListOutputTests(unittest.TestCase):
    def test_list_prints_canonical_full_list(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = gate.main(["--list"])
        self.assertEqual(rc, 0)
        printed = buf.getvalue().strip().splitlines()
        expected = [gate.render(c) for c in gate.full_command_list()]
        self.assertEqual(printed, expected)

    def test_list_includes_chelis_lint_check(self):
        # Regression guard: the historical `AGENTS.md` gate omitted
        # `chelis lint --check .`. It must be in the canonical list.
        buf = io.StringIO()
        with redirect_stdout(buf):
            gate.main(["--list"])
        self.assertIn("lint --check .", buf.getvalue())

    def test_list_uses_nextest_not_cargo_test(self):
        # Regression guard: the historical `AGENTS.md` gate said
        # `cargo test --workspace` where CI runs `cargo nextest run`.
        rendered = [gate.render(c) for c in gate.full_command_list()]
        self.assertTrue(
            any(r.startswith("cargo nextest run --workspace") for r in rendered),
            f"expected a `cargo nextest run --workspace` command, got {rendered}",
        )
        self.assertNotIn("cargo test --workspace", rendered)


# A gate `run:` step is "gate-relevant" if it invokes the Rust
# toolchain (`cargo ...`) or the Chelis CLI directly (`chelis ...`).
# The `cargo run -p chelis-cli --bin chelis -- ...` form is already
# covered by the `cargo ` prefix; the bare `chelis ...` form is the
# case the original `cargo `-only filter missed (RT-2 finding). Both
# the module docstring and
# docs/investigations/test_toolchain_guards_design.md describe the lock
# as covering "every `cargo`/`chelis` invocation", so the parser must
# catch both.
_GATE_COMMAND_PREFIXES = ("cargo ", "chelis ")


def _is_gate_relevant_command(command: str) -> bool:
    """True if `command` is a `cargo` or `chelis` invocation that a gate
    job must route through `gate.py` rather than hand-inline."""
    return any(command.startswith(prefix) for prefix in _GATE_COMMAND_PREFIXES)


def _parse_ci_gate_invocations() -> dict[str, list[str]]:
    """Parse `.github/workflows/ci.yml` and return, per gate job, the
    list of `run:` command lines that invoke `cargo` or `chelis`
    (including the `cargo run ... chelis ... lint` form).

    The parser is intentionally simple line-based YAML-shape matching:
    it tracks the current `<job>:` header (two-space indent under
    `jobs:`) and collects single-line `run:` values whose command
    starts with `cargo ` or `chelis ` (see `_is_gate_relevant_command`).
    Multi-line `run: |` blocks in the gate jobs are not used today; if
    one is introduced the parity test will not see it, which the
    `test_no_multiline_run_in_gate_jobs` guard catches.
    """
    text = CI_YML.read_text()
    lines = text.splitlines()
    current_job: str | None = None
    invocations: dict[str, list[str]] = {}
    job_header = re.compile(r"^  ([a-z0-9-]+):\s*$")
    run_inline = re.compile(r"^\s*run:\s*(.+?)\s*$")
    for line in lines:
        m = job_header.match(line)
        if m is not None:
            current_job = m.group(1)
            invocations.setdefault(current_job, [])
            continue
        if current_job is None:
            continue
        rm = run_inline.match(line)
        if rm is None:
            continue
        command = rm.group(1).strip()
        if command == "|":
            # Multi-line block; record a sentinel so the dedicated
            # guard test can detect it.
            invocations[current_job].append("<multiline-run-block>")
            continue
        if _is_gate_relevant_command(command):
            invocations[current_job].append(command)
    return invocations


def _extract_ci_bash_array(name: str) -> list[str]:
    """Extract a simple Bash array from `.github/workflows/ci.yml`.

    The no-ai-authorship job stores its grep patterns as single-quoted
    array entries. Keep this parser narrow so workflow shape changes are
    surfaced by the tests instead of silently accepted.
    """
    lines = CI_YML.read_text().splitlines()
    values: list[str] = []
    inside = False
    start = re.compile(rf"^\s*{re.escape(name)}=\(\s*$")
    entry = re.compile(r"^\s*'(.+)'\s*$")
    for line in lines:
        if not inside:
            if start.match(line):
                inside = True
            continue
        if line.strip() == ")":
            return values
        m = entry.match(line)
        if m is None:
            raise AssertionError(f"unsupported {name} array line: {line!r}")
        values.append(m.group(1))
    raise AssertionError(f"missing Bash array {name} in {CI_YML}")


class CiParityTests(unittest.TestCase):
    """The lock: every cargo/chelis gate invocation in the CI workflow
    must be produced by `gate.py`. If a future edit hand-inlines a
    cargo command into the `lint-and-unit` or `integration` job, this
    test fails."""

    def test_ci_file_exists(self):
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")

    def test_gate_jobs_call_gate_py(self):
        # The `lint-and-unit` and `integration` jobs must invoke
        # `python3 scripts/gate.py <stage>` and must NOT hand-inline
        # any `cargo` or `chelis` command.
        invocations = _parse_ci_gate_invocations()
        for job in ("lint-and-unit", "integration"):
            self.assertIn(job, invocations, f"CI job '{job}' not found")
            self.assertEqual(
                invocations[job],
                [],
                (
                    f"CI job '{job}' hand-inlines cargo/chelis command(s) "
                    f"{invocations[job]}; route them through "
                    f"scripts/gate.py instead"
                ),
            )
        # Positive parity: the workflow text must actually call
        # gate.py for both stages.
        text = CI_YML.read_text()
        self.assertIn("scripts/gate.py lint-and-unit", text)
        self.assertIn("scripts/gate.py integration", text)

    def test_non_gate_jobs_are_excluded_by_name(self):
        # The non-gate jobs are allowed to keep their own cargo/chelis
        # invocations. This test pins the exclusion list so it stays
        # visible: if a new non-gate job is added, the author must
        # decide explicitly whether it is in scope.
        invocations = _parse_ci_gate_invocations()
        present_jobs = set(invocations.keys())
        for job in NON_GATE_JOBS:
            self.assertIn(
                job,
                present_jobs,
                (
                    f"expected non-gate job '{job}' in ci.yml; if it was "
                    f"renamed or removed, update NON_GATE_JOBS"
                ),
            )

    def test_no_multiline_run_in_gate_jobs(self):
        # A `run: |` block in a gate job would hide its commands from
        # the line-based parity parser. Disallow it for the two gate
        # jobs so parity stays enforceable.
        invocations = _parse_ci_gate_invocations()
        for job in ("lint-and-unit", "integration"):
            self.assertNotIn(
                "<multiline-run-block>",
                invocations.get(job, []),
                (
                    f"CI gate job '{job}' uses a multi-line run block; "
                    f"keep gate steps as single-line `run: python3 "
                    f"scripts/gate.py ...` so the parity parser sees them"
                ),
            )


class NoAiAuthorshipTests(unittest.TestCase):
    def test_kiro_is_banned_in_authorship_patterns(self):
        message_patterns = _extract_ci_bash_array("AI_MSG_PATTERNS")
        identity_patterns = _extract_ci_bash_array("AI_IDENTITY_PATTERNS")
        all_patterns = message_patterns + identity_patterns
        self.assertTrue(
            all_patterns,
            "expected no-ai-authorship patterns in ci.yml",
        )
        self.assertTrue(
            any("kiro" in pattern.lower() for pattern in all_patterns),
            f"Kiro missing from no-ai-authorship patterns: {all_patterns}",
        )

    def test_kiro_authorship_examples_match_workflow_patterns(self):
        message_patterns = _extract_ci_bash_array("AI_MSG_PATTERNS")
        identity_patterns = _extract_ci_bash_array("AI_IDENTITY_PATTERNS")
        message_examples = [
            "Co-Authored-By: Kiro <kiro@example.invalid>",
            "Generated by Kiro",
        ]
        identity_examples = [
            "Kiro <kiro@example.invalid>",
            "Jane Kiro <jane@example.invalid>",
        ]
        for example in message_examples:
            self.assertTrue(
                any(
                    re.search(pattern, example, re.IGNORECASE)
                    for pattern in message_patterns
                ),
                f"message example was not banned by AI_MSG_PATTERNS: {example!r}",
            )
        for example in identity_examples:
            self.assertTrue(
                any(
                    re.search(pattern, example, re.IGNORECASE)
                    for pattern in identity_patterns
                ),
                f"identity example was not banned by AI_IDENTITY_PATTERNS: {example!r}",
            )


if __name__ == "__main__":
    unittest.main()
