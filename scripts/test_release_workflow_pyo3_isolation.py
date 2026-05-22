"""Guard that release.yml's `cargo build -p <target>` lines don't pull pyo3.

`release.yml` (the tag-triggered release workflow) builds `chelis-cli` and
`chelis-runtime` as release binaries. PR #184 added `.cargo/config.toml`'s
`PYO3_PYTHON=.venv/bin/python` setting and the per-job `Install uv` +
`scripts/ci_setup_uv_python.py` steps in `ci.yml` / `heavy-e2e.yml`, but
**not** in `release.yml`. The argument for keeping `release.yml` uv-free
is that its targets don't transitively pull pyo3, so pyo3-build-config
never runs there.

If a future change adds `chelis-python` (or any other pyo3-pulling crate)
as a dep of `chelis-cli` or `chelis-runtime`, that argument breaks: the
tag-push release build would fail at link time, and the breakage would
be invisible until the next release tag.

This test asserts the invariant by parsing `release.yml` for `cargo build -p X`
lines and confirming `cargo tree -p X` reports no pyo3-dep for each X.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def extract_release_targets(workflow_path: Path) -> set[str]:
    """Parse `cargo build [--release] -p <name>` invocations and return target names."""
    text = workflow_path.read_text(encoding="utf-8")
    pattern = re.compile(r"cargo\s+build(?:\s+--release)?\s+-p\s+([A-Za-z0-9_-]+)")
    return set(pattern.findall(text))


def crate_has_pyo3_dep(crate: str) -> bool:
    """Return True if `cargo tree -p <crate>` contains a pyo3 dependency."""
    result = subprocess.run(
        ["cargo", "tree", "-p", crate, "--edges", "normal", "--prefix", "none"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return any(
        line.split()[0].startswith(("pyo3", "chelis-python"))
        for line in result.stdout.splitlines()
        if line.strip()
    )


class ReleaseWorkflowPyo3IsolationTest(unittest.TestCase):
    def test_release_workflow_exists(self) -> None:
        self.assertTrue(RELEASE_WORKFLOW.is_file(), f"{RELEASE_WORKFLOW} not found")

    def test_extract_release_targets_finds_expected_set(self) -> None:
        # Sanity check the parser: at time of writing, release.yml builds
        # exactly chelis-cli + chelis-runtime. If a target is added, this
        # test will fail and the new target must be vetted for pyo3 by
        # the no-pyo3 test below.
        targets = extract_release_targets(RELEASE_WORKFLOW)
        self.assertIn("chelis-cli", targets)
        self.assertIn("chelis-runtime", targets)

    def test_no_release_target_pulls_pyo3(self) -> None:
        targets = extract_release_targets(RELEASE_WORKFLOW)
        self.assertGreater(len(targets), 0, "release.yml has no `cargo build -p` lines to check")
        for crate in sorted(targets):
            with self.subTest(crate=crate):
                self.assertFalse(
                    crate_has_pyo3_dep(crate),
                    f"`cargo tree -p {crate}` reports a pyo3 dep. "
                    "release.yml relies on these targets being pyo3-free; "
                    "either remove pyo3 from the dep graph, or add the uv "
                    "setup (Install uv + scripts/ci_setup_uv_python.py) to "
                    "release.yml's jobs.",
                )


if __name__ == "__main__":
    unittest.main()
