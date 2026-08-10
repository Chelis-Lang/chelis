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


def job_block(text: str, name: str) -> str:
    jobs = re.search(r"(?m)^jobs:\s*$", text)
    if jobs is None:
        raise AssertionError("the workflow does not contain a jobs map")
    body = text[jobs.end() :]
    headers = list(re.finditer(r"(?m)^  (?P<name>[A-Za-z0-9_-]+):\s*$", body))
    blocks = {
        header.group("name"): body[
            header.start() : headers[index + 1].start()
            if index + 1 < len(headers)
            else len(body)
        ]
        for index, header in enumerate(headers)
    }
    if name not in blocks:
        raise AssertionError(f"the workflow does not contain job {name!r}")
    return blocks[name]


def step_block(job: str, name: str) -> str:
    start = re.search(rf"(?m)^      - name: {re.escape(name)}\s*$", job)
    if start is None:
        raise AssertionError(f"the job does not contain step {name!r}")
    next_step = re.search(r"(?m)^      - ", job[start.end() :])
    end = start.end() + next_step.start() if next_step is not None else len(job)
    return job[start.start() : end]


def matrix_repos(block: str) -> set[str]:
    return set(
        re.findall(r"(?m)^          - repo: (?P<repo>[A-Za-z0-9_-]+)\s*$", block)
    )


def assert_drift_partition_contract(text: str) -> None:
    host = matrix_repos(job_block(text, "drift"))
    source = matrix_repos(job_block(text, "drift-source"))
    expected_host = {
        "nautilus",
        "coral",
        "shoals",
        "school",
        "hull",
        "whale",
        "c-earchin",
        "hello-chelis",
    }
    expected_source = {"octant", "calcify", "hydronnx"}
    if host != expected_host:
        raise AssertionError(f"the off-Nix consumer set differs: {sorted(host)}")
    if source != expected_source:
        raise AssertionError(
            f"the source-dependent consumer set differs: {sorted(source)}"
        )


def _contains_active_line(block: str, marker: str) -> bool:
    return any(
        marker in line and not line.lstrip().startswith("#")
        for line in block.splitlines()
    )


def assert_runtime_classifier_contract(text: str) -> None:
    build = job_block(text, "build-chelis")
    host = job_block(text, "drift")
    source = job_block(text, "drift-source")
    script = "ci_detect_chelis_path_deps.py"
    stage = step_block(build, "Stage toolchain tarball")
    host_classifier = step_block(
        host, "Verify an off-Nix binary consumer has no Chelis source dependency"
    )
    source_classifier = step_block(
        source, "Verify the consumer uses Chelis source dependencies"
    )
    host_report = step_block(host, "File/update drift tracking issue")
    source_report = step_block(source, "File/update drift tracking issue")

    required = (
        (stage, f'cp scripts/{script} "$staging/scripts/"'),
        (host_classifier, "id: classify"),
        (host_classifier, f'python3 "$CHELIS_TOOLCHAIN/scripts/{script}"'),
        (host_classifier, "--expect absent"),
        (source_classifier, "id: classify"),
        (source_classifier, "devenv-retry --profile ci shell --no-tui -- python"),
        (source_classifier, script),
        (source_classifier, '"$GITHUB_WORKSPACE/shell" "$GITHUB_WORKSPACE/chelis"'),
        (source_classifier, "--expect present"),
        (host_report, "if: always() && steps.classify.outcome == 'success'"),
        (source_report, "if: always() && steps.classify.outcome == 'success'"),
    )
    guarded_classifier = any(
        re.search(r"(?m)^        if:", classifier) is not None
        for classifier in (host_classifier, source_classifier)
    )
    wrong_order = host.find(host_classifier) > host.find(
        "- name: Cargo gate"
    ) or source.find(source_classifier) > source.find("- name: Cargo gate")
    if (
        any(not _contains_active_line(block, marker) for block, marker in required)
        or guarded_classifier
        or wrong_order
    ):
        raise AssertionError("the runtime source-dependency classifier differs")


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

    def test_source_dependent_cargo_legs_use_project_devenv(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert_drift_partition_contract(text)
        assert_runtime_classifier_contract(text)
        source = job_block(text, "drift-source")

        for marker in (
            "Chelis-Lang/ci/actions/setup-devenv@",
            "Chelis-Lang/ci/actions/authenticate-private-ci-input@",
            "shell: devenv-ci bash --noprofile --norc -e -o pipefail {0}",
            "devenv-retry --profile ci shell --no-tui --",
            "Checkout chelis HEAD as sibling (cargo path dep)",
        ):
            self.assertIn(marker, source)

        for marker in (
            "dtolnay/rust-toolchain",
            "astral-sh/setup-uv",
            "sudo apt-get",
            "scripts/ci_setup_uv_python.py",
        ):
            self.assertNotIn(marker, source)

    def test_binary_consumer_legs_remain_off_nix(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert_drift_partition_contract(text)
        assert_runtime_classifier_contract(text)

    def test_a_missing_runtime_classifier_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace("--expect absent", "--expect omitted", 1)
        with self.assertRaisesRegex(AssertionError, "runtime source-dependency"):
            assert_runtime_classifier_contract(mutated)

    def test_a_commented_runtime_classifier_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            '          python3 "$CHELIS_TOOLCHAIN/scripts/ci_detect_chelis_path_deps.py" \\\n',
            '          # python3 "$CHELIS_TOOLCHAIN/scripts/ci_detect_chelis_path_deps.py" \\\n',
            1,
        )
        with self.assertRaisesRegex(AssertionError, "runtime source-dependency"):
            assert_runtime_classifier_contract(mutated)

    def test_a_guarded_runtime_classifier_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            "      - name: Verify an off-Nix binary consumer has no Chelis source dependency\n",
            "      - name: Verify an off-Nix binary consumer has no Chelis source dependency\n"
            "        if: false\n",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "runtime source-dependency"):
            assert_runtime_classifier_contract(mutated)

    def test_a_classifier_after_the_cargo_gate_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        host = job_block(text, "drift")
        classifier = step_block(
            host, "Verify an off-Nix binary consumer has no Chelis source dependency"
        )
        cargo_gate = step_block(host, "Cargo gate (${{ matrix.repo }} vs HEAD)")
        mutated_host = host.replace(classifier, "", 1).replace(
            cargo_gate, cargo_gate + classifier, 1
        )
        mutated = text.replace(host, mutated_host, 1)
        with self.assertRaisesRegex(AssertionError, "runtime source-dependency"):
            assert_runtime_classifier_contract(mutated)

    def test_a_classifier_failure_does_not_report_shell_drift(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            "if: always() && steps.classify.outcome == 'success'",
            "if: always()",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "runtime source-dependency"):
            assert_runtime_classifier_contract(mutated)

    def test_a_source_dependent_leg_in_the_host_job_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            "          - repo: c-earchin", "          - repo: octant", 1
        )
        with self.assertRaisesRegex(AssertionError, "off-Nix consumer set differs"):
            assert_drift_partition_contract(mutated)

    def test_a_host_consumer_in_the_source_job_fails(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        mutated = text.replace(
            "          - repo: hydronnx", "          - repo: c-earchin", 1
        )
        with self.assertRaisesRegex(
            AssertionError, "source-dependent consumer set differs"
        ):
            assert_drift_partition_contract(mutated)

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
        )[1].split("# Host Cargo consumer only", 1)[0]

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
