"""Regression checks for the scheduled ecosystem drift workflow."""

from pathlib import Path
import re
import unittest


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "ecosystem-drift.yml"
)


def assert_devenv_rust_version_contract(text: str) -> None:
    build, drift = text.split("  build-chelis:", 1)[1].split("\n  drift:", 1)
    if re.search(r"(?m)^\s+toolchain:\s+\d+\.\d+\.\d+\s*$", drift):
        raise AssertionError("the drift job must not hardcode a Rust version")
    required = (
        "id: rust-version",
        "devenv-retry --profile ci shell --no-tui -- rustc --version",
        "rust_version: ${{ steps.rust-version.outputs.rust_version }}",
        "toolchain: ${{ needs.build-chelis.outputs.rust_version }}",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise AssertionError(f"incomplete Devenv Rust version contract: {missing!r}")
    if "id: rust-version" not in build:
        raise AssertionError("the build job must export the Devenv Rust version")


class EcosystemDriftWorkflowTests(unittest.TestCase):
    def _build_job_block(self) -> str:
        text = WORKFLOW.read_text(encoding="utf-8")
        return text.split("  build-chelis:", 1)[1].split("\n  drift:", 1)[0]

    def test_build_job_ships_the_portable_release_output(self) -> None:
        block = self._build_job_block()
        self.assertIn(
            "devenv-retry build --no-tui --quiet outputs.release-chelis", block
        )
        self.assertIn(
            "devenv-retry --profile ci shell --no-tui -- "
            "python scripts/verify_release_chelis.py",
            block,
        )
        self.assertIn("--platform linux-x86_64", block)
        self.assertIn("reef build packages/chelis-std", block)

    def test_build_job_uses_no_host_toolchain(self) -> None:
        block = self._build_job_block()
        for marker in (
            "dtolnay/rust-toolchain",
            "astral-sh/setup-uv",
            "scripts/ci_apt_get.py",
            "scripts/ci_setup_uv_python.py",
            "Swatinem/rust-cache",
            "cargo build --release",
            "strip target/release/chelis",
        ):
            self.assertNotIn(marker, block)

    def test_devenv_supplies_the_cargo_leg_rust_version(self) -> None:
        assert_devenv_rust_version_contract(WORKFLOW.read_text(encoding="utf-8"))

    def test_a_hardcoded_cargo_leg_rust_version_fails_the_contract(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            "toolchain: ${{ needs.build-chelis.outputs.rust_version }}",
            "toolchain: 1.95.0",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "must not hardcode"):
            assert_devenv_rust_version_contract(mutated)

    def test_build_job_stages_the_unpacked_release_tree(self) -> None:
        block = self._build_job_block()
        self.assertIn('cp "$HEAD_TOOLCHAIN/bin/chelis" "$staging/bin/"', block)
        self.assertIn(
            'cp "$HEAD_TOOLCHAIN/lib/libchelis_runtime.a" "$staging/lib/"', block
        )
        self.assertIn('cp "$HEAD_TOOLCHAIN"/include/*.h "$staging/include/"', block)
        self.assertNotIn("cp target/release/chelis", block)

    def test_reef_canary_raises_the_whole_suite_timeout(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        reef_test_step = text.split(
            "- name: chelis test tests/ (${{ matrix.repo }} vs HEAD)", 1
        )[1].split("# cargo legs only", 1)[0]

        self.assertIn(
            "chelis test tests/ --timeout 600 --suite-timeout 900 --jobs auto",
            reef_test_step,
        )
        self.assertEqual(reef_test_step.count("--timeout 600"), 1)
        self.assertEqual(reef_test_step.count("--suite-timeout 900"), 1)

    def test_hello_head_canary_does_not_compare_release_generated_bytes(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        hello_step = text.split(
            "- name: hello-chelis check + test + C-backend (vs HEAD)", 1
        )[1].split("# ADVISORY conformance signal", 1)[0]

        self.assertIn("run chelis check", hello_step)
        self.assertIn("run timeout 900s chelis test", hello_step)
        self.assertIn("tests/test_c_backend.py", hello_step)
        self.assertNotIn("regen_deep.py", hello_step)


if __name__ == "__main__":
    unittest.main()
