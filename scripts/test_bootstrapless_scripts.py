"""Lock the two deliberately bootstrap-free `python3` scripts.

Run via: `python3 -m unittest scripts.test_bootstrapless_scripts` from repo
root, or `python3 scripts/test_bootstrapless_scripts.py`.

`AGENTS.md` "Scripting Language Policy" says every script uses a uv-managed
interpreter, and that `scripts/gate.py` is the one `python3` entry point that
self-heals by re-executing through uv. Two process-hygiene diagnostics are
carved out of that rule: `reap_orphans.py` and `preflight_exec_probe.py` are
documented with a bare `python3` because they run before, and independently
of, a working project environment.

That carve-out is only safe while both scripts actually run on the oldest
system Python a supported workstation ships. macOS still ships 3.9, so this
module locks:

  (a) both parse as valid 3.9 syntax (`ast.parse(feature_version=(3, 9))`);
  (b) both carry `from __future__ import annotations`, which is what makes
      PEP 585/604 annotations legal on 3.9;
  (c) both import standard-library modules only, since no project
      environment is guaranteed to exist when they run;
  (d) the carve-out list here and the one named in `AGENTS.md` agree, so a
      third exempt script cannot be added in only one of the two places;
  (e) the negative control: `gate.py` still re-execs through uv, so the
      exemption stays a pair and does not quietly become the rule.
"""

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# The closed carve-out. Adding a name here without amending AGENTS.md fails
# test_agents_md_names_the_same_scripts, and vice versa.
BOOTSTRAPLESS = ("reap_orphans.py", "preflight_exec_probe.py")

OLDEST_SUPPORTED_SYSTEM_PYTHON = (3, 9)


def _source(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _toplevel_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import; not a distribution dependency
                continue
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


class BootstraplessScriptsTest(unittest.TestCase):
    def test_scripts_exist(self) -> None:
        for name in BOOTSTRAPLESS:
            self.assertTrue((SCRIPTS / name).is_file(), f"missing scripts/{name}")

    def test_parse_on_oldest_supported_system_python(self) -> None:
        for name in BOOTSTRAPLESS:
            with self.subTest(script=name):
                try:
                    ast.parse(
                        _source(name),
                        filename=name,
                        feature_version=OLDEST_SUPPORTED_SYSTEM_PYTHON,
                    )
                except SyntaxError as exc:
                    self.fail(
                        f"scripts/{name} uses syntax newer than Python "
                        f"{'.'.join(map(str, OLDEST_SUPPORTED_SYSTEM_PYTHON))}: "
                        f"{exc}. AGENTS.md documents this script with a bare "
                        f"`python3`, which on macOS is still 3.9. Either keep it "
                        f"3.9-parseable or give it gate.py's uv re-exec bootstrap "
                        f"and drop it from AGENTS.md's carve-out."
                    )

    def test_future_annotations_header(self) -> None:
        for name in BOOTSTRAPLESS:
            with self.subTest(script=name):
                tree = ast.parse(_source(name), filename=name)
                futures = {
                    alias.name
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__"
                    for alias in node.names
                }
                self.assertIn(
                    "annotations",
                    futures,
                    f"scripts/{name} must start with "
                    f"`from __future__ import annotations`; without it, PEP 585 "
                    f"and PEP 604 annotations raise at import time on 3.9.",
                )

    def test_standard_library_only(self) -> None:
        stdlib = sys.stdlib_module_names
        for name in BOOTSTRAPLESS:
            with self.subTest(script=name):
                tree = ast.parse(_source(name), filename=name)
                foreign = sorted(m for m in _toplevel_modules(tree) if m not in stdlib)
                self.assertEqual(
                    [],
                    foreign,
                    f"scripts/{name} imports non-stdlib module(s) {foreign}. It is "
                    f"documented as a bare `python3` invocation, so no project "
                    f"environment is guaranteed to be installed when it runs.",
                )

    def test_agents_md_names_the_same_scripts(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        marker = "Two diagnostics are deliberately bootstrap-free."
        self.assertIn(
            marker,
            agents,
            "AGENTS.md no longer documents the bootstrap-free carve-out; "
            "update this test and the policy together.",
        )
        for name in BOOTSTRAPLESS:
            self.assertIn(
                f"`scripts/{name}`",
                agents,
                f"scripts/{name} is exempt here but AGENTS.md does not say so.",
            )

    def test_gate_still_bootstraps(self) -> None:
        """Negative control: the exemption must not spread to gate.py."""
        gate = _source("gate.py")
        self.assertIn(
            "def ensure_managed_runtime(",
            gate,
            "gate.py lost its uv re-exec bootstrap; AGENTS.md promises that a "
            "bare `python3 scripts/gate.py` is always safe.",
        )


if __name__ == "__main__":
    unittest.main()
