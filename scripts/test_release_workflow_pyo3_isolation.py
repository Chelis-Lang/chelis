"""Guard that release.yml's `cargo build -p <target>` lines don't pull pyo3.

The shipped CLI and runtime must remain independent of Python bindings.
Release jobs provision managed Python solely to observe native Cargo builds;
that build-time interpreter does not authorize a PyO3 runtime dependency.

This test asserts the invariant by parsing `release.yml` for `cargo build -p X`
lines and confirming `cargo tree -p X` reports no pyo3-dep for each X.
"""

from __future__ import annotations

import re
import shutil
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


def crate_has_pyo3_dep(crate: str, *, features: str | None = None) -> bool:
    """Return True if `cargo tree -p <crate>` contains a pyo3 dependency.

    `features`, when given, is passed as `--features <features>` so the
    feature-enabled dep graph is vetted too. chelis#422 (WS-4) makes
    release.yml build `chelis-cli --features smt`; the smt feature pulls
    cvc5 (which does not link pyo3), but a future feature edit could, so
    the smt graph is vetted alongside the default one.
    """
    cmd = ["cargo", "tree", "-p", crate, "--edges", "normal", "--prefix", "none"]
    if features:
        cmd += ["--features", features]
    result = subprocess.run(
        cmd,
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
    def test_extract_release_targets_finds_expected_set(self) -> None:
        # Sanity check the parser: release.yml builds chelis-cli +
        # chelis-runtime (the toolchain tarball) and chelisup (the bare
        # `chelisup-<slug>` bootstrap asset, WS-B / chelis#164). If a
        # target is added, this test will fail and the new target must be
        # vetted for pyo3 by the no-pyo3 test below.
        targets = extract_release_targets(RELEASE_WORKFLOW)
        self.assertIn("chelis-cli", targets)
        self.assertIn("chelis-runtime", targets)
        self.assertIn("chelisup", targets)

    @unittest.skipUnless(
        shutil.which("cargo") is not None,
        "cargo unavailable; skipping the `cargo tree` pyo3 dep-graph oracle",
    )
    def test_no_release_target_pulls_pyo3(self) -> None:
        targets = extract_release_targets(RELEASE_WORKFLOW)
        self.assertGreater(len(targets), 0, "release.yml has no `cargo build -p` lines to check")
        for crate in sorted(targets):
            with self.subTest(crate=crate):
                self.assertFalse(
                    crate_has_pyo3_dep(crate),
                    f"`cargo tree -p {crate}` reports a pyo3 dep. "
                    "release targets must stay PyO3-free; remove the Python "
                    "binding dependency from the shipping graph.",
                )

    @unittest.skipUnless(
        shutil.which("cargo") is not None,
        "cargo unavailable; skipping the `cargo tree` pyo3 dep-graph oracle",
    )
    def test_chelis_cli_smt_feature_is_pyo3_free(self) -> None:
        # chelis#422 (WS-4): release.yml builds `chelis-cli --features
        # smt`. The smt feature pulls cvc5 (no pyo3) plus chelis-tide's
        # smt forward, but a future feature edit could introduce a
        # pyo3-pulling crate, violating the shipping dependency contract.
        # Vet the smt graph explicitly, not just the default one above.
        if "chelis-cli" not in extract_release_targets(RELEASE_WORKFLOW):
            self.skipTest("release.yml does not build chelis-cli")
        self.assertFalse(
            crate_has_pyo3_dep("chelis-cli", features="smt"),
            "`cargo tree -p chelis-cli --features smt` reports a pyo3 dep. "
            "the shipping CLI smt graph must remain PyO3-free.",
        )


if __name__ == "__main__":
    unittest.main()
