"""Guard that release.yml's `cargo build -p <target>` lines don't pull pyo3.

`release.yml` builds `chelis-cli` and `chelis-runtime` as release binaries.
The portable `chelisup` path uses the Devenv output instead of host Cargo.
PR #184 added `.cargo/config.toml`'s
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
    def test_release_workflow_exists(self) -> None:
        self.assertTrue(RELEASE_WORKFLOW.is_file(), f"{RELEASE_WORKFLOW} not found")

    def test_extract_release_targets_finds_expected_set(self) -> None:
        # The full toolchain jobs build chelis-cli and chelis-runtime.
        # Devenv builds the portable chelisup output from committed Cargo.nix.
        # If a host Cargo target appears, the no-pyo3 test covers that target.
        targets = extract_release_targets(RELEASE_WORKFLOW)
        self.assertEqual(targets, {"chelis-cli", "chelis-runtime"})

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
                    "release.yml relies on these targets being pyo3-free; "
                    "either remove pyo3 from the dep graph, or add the uv "
                    "setup (Install uv + scripts/ci_setup_uv_python.py) to "
                    "release.yml's jobs.",
                )

    @unittest.skipUnless(
        shutil.which("cargo") is not None,
        "cargo unavailable; skipping the `cargo tree` pyo3 dep-graph oracle",
    )
    def test_chelis_cli_smt_feature_is_pyo3_free(self) -> None:
        # chelis#422 (WS-4): release.yml builds `chelis-cli --features
        # smt`. The smt feature pulls cvc5 (no pyo3) plus chelis-tide's
        # smt forward, but a future feature edit could introduce a
        # pyo3-pulling crate, which would break the uv-free release build
        # at link time. Vet the smt graph explicitly, not just the
        # default one above.
        if "chelis-cli" not in extract_release_targets(RELEASE_WORKFLOW):
            self.skipTest("release.yml does not build chelis-cli")
        self.assertFalse(
            crate_has_pyo3_dep("chelis-cli", features="smt"),
            "`cargo tree -p chelis-cli --features smt` reports a pyo3 dep. "
            "release.yml builds chelis-cli --features smt uv-free; either "
            "remove pyo3 from the smt dep graph, or add the uv setup to "
            "release.yml's jobs.",
        )


if __name__ == "__main__":
    unittest.main()
