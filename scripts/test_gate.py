"""Unit tests for `gate.py`.

Run via:
`uv run --managed-python --python 3.11 --no-project python -m unittest
scripts.test_gate` from the repo root.

Four things are locked here:

  (a) the full developer gate keeps the complete default nextest profile while
      CI delegates its two census binaries from the `ci` profile to the dtype
      oracle;
  (b) a parity assertion: every `cargo`/`chelis` invocation in a gate
      step of `.github/workflows/ci.yml` is produced by `gate.py`. This
      covers both `cargo ...` and bare `chelis ...` commands (the
      `cargo run -p chelis-cli --bin chelis -- ...` form is caught by
      the `cargo ` prefix). This is the lock that turns future
      CI-vs-gate drift into a test failure. The non-gate jobs
      (dtype oracle/aggregator, sanitizer, macOS-smoke, docs, LOC-report,
      no-AI-authorship) are excluded by name so the exclusion is explicit
      and reviewable;
  (c) `--list` prints the canonical full list;
  (d) no-ai-authorship patterns cover the current banned tool identities.
"""

import importlib.util
import io
import re
import shlex
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
SPAN_COMMENTS_RS = (
    REPO_ROOT / "crates" / "chelis-backend-c" / "tests" / "span_comments.rs"
)
SMT_FULL_PROVE_YML = REPO_ROOT / ".github" / "workflows" / "smt-full-prove.yml"
CHELIS_PROVE_TOML = REPO_ROOT / "crates" / "chelis-prove" / "Cargo.toml"
NIX_PACKAGES_YML = REPO_ROOT / ".github" / "workflows" / "nix-packages.yml"
RELEASE_YML = REPO_ROOT / ".github" / "workflows" / "release.yml"
DEVENV_TOOLCHAINS_NIX = REPO_ROOT / "devenv" / "toolchains.nix"
DEVENV_SETUP_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@111b5865ccf04146344ad99dcdea9d724d65fc69"
)
DEVENV_AUTH_ACTION = (
    "Chelis-Lang/ci/actions/authenticate-private-ci-input@"
    "111b5865ccf04146344ad99dcdea9d724d65fc69"
)
PORTABLE_DEVENV_SHELL = "devenv-ci bash --noprofile --norc -e -o pipefail {0}"
DEVENV_COMMAND_PREFIX = "devenv-retry --profile ci shell --no-tui -- "
DEVENV_COMMAND_PREFIXES = (
    DEVENV_COMMAND_PREFIX,
    "devenv-retry --profile sanitizers shell --no-tui -- ",
    "devenv-retry --profile smt shell --no-tui -- ",
    # Keep bare wrappers visible to the parity scanner. The workflow
    # composition contract rejects them as retry bypasses.
    "devenv --profile ci shell --no-tui -- ",
    "devenv --profile sanitizers shell --no-tui -- ",
    "devenv --profile smt shell --no-tui -- ",
)
# These jobs run project commands. Portable, off-Nix, and orchestration jobs
# have closed dispositions in test_ci_execution_ownership.py.
PROJECT_DEVENV_WORKFLOW_JOBS = {
    "ci.yml": (
        "diagnostic-kind-oracle",
        "rejection-authority-liveness",
        "lint-and-unit",
        "workspace-tests",
        "dtype-phase3-oracle",
        "faithful-observation-phase2-oracle",
        "macos-smoke",
        "backend-sanitizers",
        "docs",
        "smt-build",
        "smt-build-darwin-arm64",
    ),
    "conformance.yml": ("conformance",),
    "conformance-nightly.yml": ("conformance-nightly",),
    "heavy-e2e.yml": ("heavy-e2e",),
    "smt-full-prove.yml": ("full-smt-prove",),
    "ecosystem-drift.yml": ("build-chelis", "drift-source"),
    "loc-report.yml": ("loc-report",),
}
DOCS_ONLY_GATE_IF = (
    "if: ${{ !cancelled() && (needs.changes.result != 'success' "
    "|| needs.changes.outputs.docs_only != 'true') }}"
)
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CARCARA_FULL_SUITE_COMMAND = (
    "devenv-retry --profile smt shell --no-tui -- "
    "cargo test -p chelis-prove --features carcara -- --test-threads=1"
)


def _nix_supported_systems(contracts: str) -> set[str]:
    supported_block = re.search(
        r"supportedSystems\s*=\s*\[(?P<body>.*?)\];",
        contracts,
        re.DOTALL,
    )
    if supported_block is None:
        raise AssertionError("missing supportedSystems contract")
    return set(re.findall(r'"([^"]+)"', supported_block.group("body")))


def _nix_native_job_systems(workflow: str) -> set[str]:
    return set(re.findall(r"name: Nix Packages \(([^)]+)\)", workflow))


def _assert_nix_system_job_parity(contracts: str, workflow: str) -> None:
    supported = _nix_supported_systems(contracts)
    native_jobs = _nix_native_job_systems(workflow)
    if supported != native_jobs:
        missing_jobs = sorted(supported - native_jobs)
        extra_jobs = sorted(native_jobs - supported)
        raise AssertionError(
            "Nix supported systems and native CI jobs differ: "
            f"missing jobs={missing_jobs}, extra jobs={extra_jobs}"
        )


def _workflow_job_blocks(workflow: str) -> dict[str, str]:
    headers = list(re.finditer(r"(?m)^  (?P<name>[a-z0-9-]+):\s*$", workflow))
    blocks: dict[str, str] = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(workflow)
        blocks[header.group("name")] = workflow[header.start() : end]
    return blocks


def _macos_manual_dispatch_errors(workflow: str) -> list[str]:
    trigger = workflow[: workflow.index("\njobs:\n")]
    has_manual_trigger = "workflow_dispatch:" in trigger
    manual_if = re.compile(
        r"(?m)^    if:\s*(?:\$\{\{\s*)?.*"
        r"github\.event_name\s*==\s*'workflow_dispatch'.*$"
    )
    errors: list[str] = []
    for job, block in _workflow_job_blocks(workflow).items():
        direct_macos = "runs-on: macos-latest" in block
        matrix_macos = (
            "runs-on: ${{ matrix.os }}" in block and "os: macos-latest" in block
        )
        if not direct_macos and not matrix_macos:
            continue
        if not has_manual_trigger:
            errors.append(f"{job}: missing workflow_dispatch trigger")
        if manual_if.search(block) is None:
            errors.append(f"{job}: missing manual-dispatch job condition")
    return errors


def _assert_carcara_full_suite_command(workflow: str) -> None:
    block = _workflow_job_blocks(workflow).get("full-smt-prove")
    if block is None:
        raise AssertionError("missing full-smt-prove job")

    def run_steps() -> list[tuple[str, bool]]:
        lines = block.splitlines()
        steps_index = next(
            (index for index, line in enumerate(lines) if line.strip() == "steps:"),
            None,
        )
        if steps_index is None:
            return []
        steps_indent = len(lines[steps_index]) - len(lines[steps_index].lstrip())
        runs: list[tuple[str, bool]] = []
        current: dict[str, str] | None = None
        step_indent = steps_indent + 2

        def finish_step() -> None:
            if current is not None and "run" in current:
                runs.append((current["run"].strip(), "if" in current))

        index = steps_index + 1
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped and not stripped.startswith("#") and indent <= steps_indent:
                break
            if indent == step_indent and stripped.startswith("- "):
                finish_step()
                current = {}
                property_text = stripped[2:]
                property_indent = step_indent
            elif current is not None and indent == step_indent + 2:
                property_text = stripped
                property_indent = step_indent + 2
            else:
                index += 1
                continue

            match = re.match(r"(?P<key>run|if):(?:\s*(?P<value>.*))?$", property_text)
            if match is None:
                index += 1
                continue
            key = match.group("key")
            value = match.group("value") or ""
            if key == "run" and re.fullmatch(r"[|>][+-]?", value):
                block_lines: list[str] = []
                index += 1
                while index < len(lines):
                    block_line = lines[index]
                    block_stripped = block_line.strip()
                    block_indent = len(block_line) - len(block_line.lstrip())
                    if block_stripped and block_indent <= property_indent:
                        break
                    block_lines.append(block_line[property_indent + 2 :])
                    index += 1
                current[key] = "\n".join(block_lines)
                continue
            current[key] = value
            index += 1
        finish_step()
        return runs

    def enables_carcara(command: str) -> bool:
        try:
            words = shlex.split(command, comments=True)
        except ValueError:
            return "features" in command or "-F" in command
        if "cargo" not in words:
            return False

        def feature_value_may_enable(value: str) -> bool:
            if "$" in value or "`" in value:
                return True
            return "carcara" in re.split(r"[\s,]+", value)

        for index, word in enumerate(words):
            if word == "--all-features":
                return True
            if word in ("--features", "-F"):
                if index + 1 >= len(words):
                    return True
                if feature_value_may_enable(words[index + 1]):
                    return True
            elif word.startswith("--features="):
                if feature_value_may_enable(word.partition("=")[2]):
                    return True
            elif word.startswith("-F") and word != "-F":
                if feature_value_may_enable(word[2:]):
                    return True
        return False

    steps = run_steps()
    carcara_steps = [
        (command, conditional)
        for command, conditional in steps
        if enables_carcara(command)
    ]
    canonical_words = shlex.split(CARCARA_FULL_SUITE_COMMAND)
    canonical_steps = [
        command
        for command, conditional in carcara_steps
        if not conditional and shlex.split(command, comments=True) == canonical_words
    ]
    if len(carcara_steps) != 1 or canonical_steps != [CARCARA_FULL_SUITE_COMMAND]:
        raise AssertionError(
            "full-smt-prove must execute the complete serialized Carcara suite "
            f"exactly once in an unconditional step; found {carcara_steps}"
        )


def _assert_carcara_feature_tree_is_gmp_only(feature_tree: str) -> None:
    forbidden = (
        'gmp-mpfr-sys feature "mpfr"',
        'gmp-mpfr-sys feature "mpc"',
        'rug feature "float"',
        'rug feature "complex"',
    )
    active = [feature for feature in forbidden if feature in feature_tree]
    if active:
        raise AssertionError(
            "Carcara feature graph must stay GMP-only; activated " + ", ".join(active)
        )


def _assert_native_devenv_recipe(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    for job in ("nix-linux-x86-64", "nix-darwin-arm64"):
        if job not in blocks:
            raise AssertionError(f"missing native Nix job {job!r}")
        block = blocks[job]
        required_markers = (
            f"uses: {DEVENV_SETUP_ACTION}",
            f"shell: {PORTABLE_DEVENV_SHELL}",
            "run: devenv-retry test --no-tui",
            (
                "run: devenv-retry build --no-tui outputs.chelis "
                "outputs.chelis-runtime outputs.chelisup"
            ),
            f"uses: {DEVENV_AUTH_ACTION}",
            "app-client-id: ${{ vars.CI_APP_ID }}",
            "app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}",
        )
        for marker in required_markers:
            actual_count = block.count(marker)
            if actual_count != 1:
                raise AssertionError(
                    f"native Devenv recipe marker {marker!r} in {job!r}: "
                    f"expected 1, found {actual_count}"
                )
        setup_index = block.index(f"uses: {DEVENV_SETUP_ACTION}")
        verify_index = block.find("name: Verify the runner system")
        if verify_index < 0 or setup_index > verify_index:
            raise AssertionError(
                f"native Devenv setup in {job!r} must precede the runner verification"
            )

    forbidden_setup_markers = (
        "uses: cachix/install-nix-action@",
        "uses: cachix/cachix-action@",
        "nix profile add github:cachix/devenv/",
        "repositories: ci",
        "access-tokens = github.com=",
    )
    for marker in forbidden_setup_markers:
        if marker in workflow:
            raise AssertionError(
                f"native Devenv recipe duplicates central setup marker {marker!r}"
            )

    forbidden_python_markers = (
        "astral-sh/setup-uv",
        "uv venv",
        ".venv/bin/python",
        ".devenv/state/venv/bin/python",
    )
    found_python = [marker for marker in forbidden_python_markers if marker in workflow]
    if found_python:
        raise AssertionError(
            f"native Devenv recipe creates a host Python environment: {found_python!r}"
        )


def _assert_nix_docs_only_gate(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    changes = blocks.get("changes", "")
    if "uses: ./.github/actions/detect-docs-only" not in changes:
        raise AssertionError(
            "the Nix workflow must compute docs_only with the shared detector"
        )
    linux = blocks.get("nix-linux-x86-64", "")
    if "needs: [changes]" not in linux or DOCS_ONLY_GATE_IF not in linux:
        raise AssertionError(
            "the Linux Nix job must skip docs-only pull requests via the "
            "shared job-level gate"
        )
    darwin = blocks.get("nix-darwin-arm64", "")
    if "needs.changes" in darwin:
        raise AssertionError(
            "the darwin Nix job must keep manual dispatch as its only gate"
        )


def _assert_darwin_manual_dispatch(workflow: str) -> None:
    trigger_section = workflow.split("jobs:", 1)[0]
    if "workflow_dispatch:" not in trigger_section:
        raise AssertionError("the Nix workflow must expose a workflow_dispatch trigger")
    blocks = _workflow_job_blocks(workflow)
    darwin = blocks.get("nix-darwin-arm64", "")
    if "if: github.event_name == 'workflow_dispatch'" not in darwin:
        raise AssertionError("the darwin Nix job must run on manual dispatch only")
    linux = blocks.get("nix-linux-x86-64", "")
    if "github.event_name" in linux:
        raise AssertionError("the Linux Nix job must keep pull request coverage")


def _assert_runner_resource_bounds(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    max_jobs = {
        "nix-linux-x86-64": "max-jobs = 2",
        "nix-darwin-arm64": "max-jobs = 1",
    }
    for job, bound in max_jobs.items():
        block = blocks.get(job, "")
        for required in ("sandbox = true", bound, "cores = 0"):
            if required not in block:
                raise AssertionError(f"{job!r} must set {required!r} in NIX_CONFIG")
    linux = blocks.get("nix-linux-x86-64", "")
    reclaim_action = (
        "Chelis-Lang/ci/actions/reclaim-ubuntu-runner-disk@"
        "111b5865ccf04146344ad99dcdea9d724d65fc69"
    )
    reclaim_index = linux.find(f"uses: {reclaim_action}")
    setup_index = linux.find(f"uses: {DEVENV_SETUP_ACTION}")
    if reclaim_index < 0 or setup_index < 0 or reclaim_index > setup_index:
        raise AssertionError(
            "the Linux Nix job must reclaim runner disk before Devenv setup"
        )
    if "sudo python3" in linux:
        raise AssertionError("the Linux Nix job retains host disk cleanup")
    darwin = blocks.get("nix-darwin-arm64", "")
    if reclaim_action in darwin:
        raise AssertionError("the Darwin Nix job must not run Ubuntu disk cleanup")


def _assert_cvc5_closure_cache(workflow: str) -> None:
    # The key derivation, archive import/export, and actions/cache calls
    # live in the shared composite pair under .github/actions/; the
    # workflow-side contract is that each native job consumes both halves
    # for its own system, restore-before / save-after the flake check.
    # test_smt_lane_workflows.py locks the composite action internals.
    blocks = _workflow_job_blocks(workflow)
    jobs = (
        ("nix-linux-x86-64", "x86_64-linux"),
        ("nix-darwin-arm64", "aarch64-darwin"),
    )
    for job, system in jobs:
        block = blocks.get(job, "")
        markers = (
            "uses: ./.github/actions/cvc5-cache-restore",
            "uses: ./.github/actions/cvc5-cache-save",
            f"system: {system}",
        )
        for marker in markers:
            if marker not in block:
                raise AssertionError(
                    f"{job!r} must cache the cvc5 closure: missing {marker!r}"
                )
        check_index = block.index("run: nix flake check")
        if block.index("uses: ./.github/actions/cvc5-cache-restore") > check_index:
            raise AssertionError(
                f"{job!r} must restore the cvc5 closure before the flake check"
            )
        if block.index("uses: ./.github/actions/cvc5-cache-save") < check_index:
            raise AssertionError(
                f"{job!r} must save the cvc5 closure after the flake check"
            )


# CI jobs that are deliberately NOT part of the per-PR developer gate.
# `gate.py` only owns the `lint-and-unit` and `workspace-tests` jobs; these
# are listed by name so the parity test's exclusion is visible.
NON_GATE_JOBS = {
    "macos-smoke",
    "backend-sanitizers",
    "no-ai-authorship",
    "docs",
    # Rule-id: GATE-SCOPE-CHANGES -- the changes job computes the
    # docs_only output that gates the heavy jobs' `if` (chelis#419). It
    # runs scripts/ci_detect_docs_only.py, no cargo/chelis command, so it
    # is out of gate.py scope by design.
    "changes",
    # Rule-id: GATE-SCOPE-REJECTION-AUTHORITY -- the network-backed
    # [05-UNS-5] manifest liveness check is CI-owned and deliberately absent
    # from the offline developer gate.
    "rejection-authority-liveness",
    # Rule-id: GATE-SCOPE-DIAGNOSTIC-KIND -- C2.2's controlled source
    # mutations are CI-owned and run only when an owner/control path changes.
    "diagnostic-kind-oracle",
    # Rule-id: GATE-SCOPE-DTYPE-ORACLE -- the authoritative Phase 0-3
    # acceptance driver is CI-owned. It runs beside the workspace suite,
    # while the integration job below aggregates both outcomes under the
    # stable branch-protection context.
    "dtype-phase3-oracle",
    # Rule-id: GATE-SCOPE-FAITHFUL-OBSERVATION-ORACLE -- the authoritative
    # #732 Phase 2 acceptance driver is a dedicated CI job. It runs beside
    # the workspace and dtype legs and is aggregated under the stable
    # branch-protection context.
    "faithful-observation-phase2-oracle",
    "integration",
    # Rule-id: GATE-SCOPE-SMT -- the smt-build job is the required fast
    # cvc5-backed `smt` feature smoke. It is out of gate.py scope by
    # design, like backend-sanitizers; the full prove corpus lives in
    # smt-full-prove.yml. Runbook:
    # docs/smt_build_setup.md.
    "smt-build",
    # Rule-id: GATE-SCOPE-SMT -- the chelis#422 prove-in-CI lane that
    # builds `chelis-cli --features smt` (cvc5 from source) on the release
    # target ubuntu's smt-build does not cover: macOS-arm64. Same
    # from-source cvc5 cost as smt-build, so out of the per-PR gate scope
    # by design. (The former glibc-2.31 debian:11 lane retired with the
    # Nix-built Linux release; see openspec switch-linux-release-to-nix.)
    "smt-build-darwin-arm64",
}


class RejectionAuthorityLivenessJobTests(unittest.TestCase):
    def test_job_is_change_gated_and_has_issue_read_access(self):
        block = _ci_job_block("rejection-authority-liveness")
        self.assertIn("needs: [changes]", block)
        self.assertIn("needs.changes.outputs.rejection_authority_changed", block)
        self.assertIn("issues: read", block)
        self.assertIn("contents: read", block)
        self.assertIn("scripts/check_rejection_authority_boundary.py", block)
        self.assertIn("scripts/validate_rejection_issue_manifest.py", block)


class DiagnosticKindOracleJobTests(unittest.TestCase):
    def test_job_is_change_gated_and_executes_the_mutation_oracle(self):
        block = _ci_job_block("diagnostic-kind-oracle")
        self.assertIn("needs: [changes]", block)
        self.assertIn("needs.changes.outputs.diagnostic_kind_changed", block)
        self.assertIn("contents: read", block)
        _assert_executable_run_once(
            block,
            "python scripts/diagnostic_kind_oracle.py",
        )
        self.assertIn(f"uses: {DEVENV_SETUP_ACTION}", block)
        self.assertNotIn("taiki-e/install-action@nextest", block)

    def test_a_quoted_passing_noop_is_not_the_oracle_step(self):
        block = _ci_job_block("diagnostic-kind-oracle")
        mutated = block.replace(
            f"run: {DEVENV_COMMAND_PREFIX}python scripts/diagnostic_kind_oracle.py",
            'run: "true # scripts/diagnostic_kind_oracle.py"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(
                mutated,
                "python scripts/diagnostic_kind_oracle.py",
            )


# Whole WORKFLOW FILES that are out-of-scope-by-design for the per-PR developer
# `gate.py` quartet (like the backend-sanitizers / macos-smoke jobs in ci.yml,
# but in their own files). They run their own commands the gate does not
# produce, by design. Listed here so the exclusion is explicit and reviewable.
# Rule-id: GATE-SCOPE-CONFORMANCE -- the Hull conformance gate runs a Python
# corpus runner against the built binary; it is a CI job, NOT part of the cargo
# quartet + lint developer gate (see tests/conformance/hull/run_conformance.py
# and .github/workflows/conformance.yml).
NON_GATE_WORKFLOWS = {
    "ci.yml",
    "smt-full-prove.yml",
    "heavy-e2e.yml",
    "release.yml",
    "conformance.yml",
    "conformance-nightly.yml",
    # The ecosystem drift canary is a scheduled cross-repo workflow that
    # builds chelis HEAD and runs each downstream shell's gate against it.
    # It runs nothing the per-PR gate owns and never runs on PR/push, so it
    # is out of the gate.py quartet scope by design.
    "ecosystem-drift.yml",
    # LOC report moved out of ci.yml's per-merge path into its own weekly
    # scheduled workflow; it commits a docs/loc_report.md bot commit and runs
    # nothing the per-PR gate owns, so it is out of gate.py scope by design.
    "loc-report.yml",
    # Native Nix package jobs build the complete flake check set on Linux and
    # macOS. These jobs prove a separate source-build channel and do not run
    # commands from the canonical Cargo gate.
    "nix-packages.yml",
    # Scheduled shared Actions-cache pruner. Deletes stale caches to hold the
    # pool under the repository limit; runs no per-PR gate
    # command. Out of gate.py scope by design.
    "cache-prune.yml",
    # OpenSpec validation uses the pinned central action in advisory mode.
    # It runs no cargo or Chelis command that the developer gate owns.
    # It stays outside gate.py by design.
    "openspec-validate.yml",
}


class StageUnionTests(unittest.TestCase):
    def test_full_gate_keeps_censuses_while_ci_uses_the_split_profile(self):
        self.assertEqual(gate.STAGES["integration"], [gate.NEXTEST_WORKSPACE_CI])
        self.assertIn(gate.NEXTEST_WORKSPACE, gate.full_command_list())
        self.assertNotIn(gate.NEXTEST_WORKSPACE_CI, gate.full_command_list())
        self.assertNotIn("--profile", gate.NEXTEST_WORKSPACE)
        self.assertIn("--no-fail-fast", gate.NEXTEST_WORKSPACE)
        self.assertEqual(
            gate.NEXTEST_WORKSPACE_CI[-3:],
            ["--profile", "ci", "--no-fail-fast"],
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
        # Each command line is `<command>  # <local-vs-ci annotation>`
        # (chelis#360); the command part must still be exactly the
        # canonical full list, in order. Standalone `#`-comment lines
        # (the --local dynamic-stage note) are not commands.
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = gate.main(["--list"])
        self.assertEqual(rc, 0)
        printed = [
            line
            for line in buf.getvalue().strip().splitlines()
            if not line.startswith("#")
        ]
        commands = [line.split("  # ")[0] for line in printed]
        expected = [gate.render(c) for c in gate.full_command_list()]
        self.assertEqual(commands, expected)

    def test_list_includes_chelis_lint_check(self):
        # Regression guard: the historical `AGENTS.md` gate omitted
        # `chelis lint --check .`. It must be in the canonical list.
        buf = io.StringIO()
        with redirect_stdout(buf):
            gate.main(["--list"])
        self.assertIn("lint --check .", buf.getvalue())

    def test_list_includes_a_doctest_stage(self):
        # Regression guard (chelis#875): `cargo nextest` does not execute
        # doctests, so a gate made entirely of nextest stages runs none of
        # the `compile_fail` oracles in crates/chelis-types/src/errors.rs.
        # Those oracles are the chelis#731 Phase 2 acceptance artifact; a
        # gate that does not drive them lets the plan claim a compile-time
        # guarantee no continuous job checks. If this stage is ever
        # removed, the oracles go dark silently -- hence an explicit lock
        # rather than relying on the union tests.
        rendered = [gate.render(c) for c in gate.full_command_list()]
        self.assertTrue(
            any(r.endswith("--doc") for r in rendered),
            f"expected a doctest stage in the canonical list, got {rendered}",
        )

    def test_doctest_stage_is_in_the_local_subset(self):
        # The doctest stage costs well under a second and catches a broken
        # oracles before push rather than in CI, so they belong in `--local`
        # too (chelis#875 and chelis#959).
        rendered = [gate.render(c) for c in gate.LOCAL_STATIC_COMMANDS]
        self.assertIn("cargo test -p chelis-types --doc", rendered)
        self.assertIn("cargo test -p chelis-compiler-api --doc", rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_lint_and_unit_stage(self):
        rendered = [gate.render(command) for command in gate.STAGES["lint-and-unit"]]
        for command in (
            "cargo test -p chelis-compiler-api --doc",
            "cargo test -p chelis-pipeline-core --doc",
            "<managed-python> scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_local_subset(self):
        rendered = [gate.render(command) for command in gate.LOCAL_STATIC_COMMANDS]
        for command in (
            "cargo test -p chelis-compiler-api --doc",
            "cargo test -p chelis-pipeline-core --doc",
            "<managed-python> scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_pipeline_core_boundary_guards_are_in_the_lint_and_unit_stage(self):
        # The dependency guard, documentation guard, and pipeline-artifact
        # compile-fail fixture must run in the per-PR gate (hosted CI runs
        # `gate.py lint-and-unit`), not only in the manual oracle.
        rendered = [gate.render(command) for command in gate.STAGES["lint-and-unit"]]
        for command in (
            "<managed-python> scripts/pipeline_core_dependency_guard.py",
            "<managed-python> scripts/pipeline_core_documentation_guard.py",
            "<managed-python> scripts/check_pipeline_core_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_cheap_pipeline_core_guards_are_in_the_local_subset(self):
        # The two cheap guards (cargo metadata + pure Python) run pre-push; the
        # out-of-workspace compile-fail build stays CI/full-gate only.
        rendered = [gate.render(command) for command in gate.LOCAL_STATIC_COMMANDS]
        for command in (
            "<managed-python> scripts/pipeline_core_dependency_guard.py",
            "<managed-python> scripts/pipeline_core_documentation_guard.py",
        ):
            self.assertIn(command, rendered)
        self.assertNotIn(
            "<managed-python> scripts/check_pipeline_core_compile_fail.py",
            rendered,
        )

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


def _unwrap_devenv_command(command: str) -> str:
    """Return the command that runs inside a single Devenv shell wrapper."""
    for prefix in DEVENV_COMMAND_PREFIXES:
        if command.startswith(prefix):
            return command.removeprefix(prefix)
    return command


def _is_gate_relevant_command(command: str) -> bool:
    """True if `command` is a `cargo` or `chelis` invocation that a gate
    job must route through `gate.py` rather than hand-inline."""
    logical_command = _unwrap_devenv_command(command)
    return any(logical_command.startswith(prefix) for prefix in _GATE_COMMAND_PREFIXES)


def _parse_ci_gate_invocations() -> dict[str, list[str]]:
    """Parse `.github/workflows/ci.yml` and return, per gate job, the
    list of `run:` command lines that invoke `cargo` or `chelis`
    (including the `cargo run ... chelis ... lint` form).

    The parser is intentionally simple line-based YAML-shape matching:
    it tracks the current `<job>:` header (two-space indent under
    `jobs:`) and collects single-line `run:` values whose logical command
    starts with `cargo ` or `chelis ` (see `_is_gate_relevant_command`).
    It removes one project Devenv shell prefix before that classification.
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
            invocations[current_job].append(_unwrap_devenv_command(command))
    return invocations


def _ci_job_block(job: str) -> str:
    """Return the raw `.github/workflows/ci.yml` text block for one job."""
    return _workflow_job_block(CI_YML, job)


def _assert_executable_run_once(job_block: str, command: str) -> None:
    """Require one executable, unquoted single-line `run:` scalar.

    Comments and quoted no-ops that merely contain `command` are deliberately
    excluded; substring counting would let either masquerade as the oracle.
    """
    run_scalar = re.compile(r"^\s+run:\s*(?P<command>\S.*?)\s*$")
    executable = []
    for line in job_block.splitlines():
        match = run_scalar.match(line)
        if match is None:
            continue
        value = match.group("command")
        if value not in {"|", ">", "|-", ">-"}:
            executable.append(_unwrap_devenv_command(value))
    count = executable.count(command)
    if count != 1:
        raise AssertionError(
            f"expected exactly one executable `run: {command}`, found {count}"
        )


def _workflow_job_block(path: Path, job: str) -> str:
    """Return the raw workflow text block for one job."""
    lines = path.read_text().splitlines()
    header = f"  {job}:"
    start: int | None = None
    for idx, line in enumerate(lines):
        if line == header:
            start = idx
            break
    if start is None:
        raise AssertionError(f"missing workflow job {job!r} in {path}")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if re.match(r"^  [a-z0-9-]+:\s*$", lines[idx]):
            end = idx
            break
    return "\n".join(lines[start:end])


def _assert_no_workflow_ci_policy(workflow: str) -> None:
    forbidden = (
        "CARGO_PROFILE_DEV_DEBUG:",
        "CARGO_PROFILE_TEST_DEBUG:",
        "CHELIS_C_TEST_EXTRA_FLAGS:",
        "ASAN_OPTIONS:",
        "UBSAN_OPTIONS:",
    )
    found = [marker for marker in forbidden if marker in workflow]
    if found:
        raise AssertionError(f"workflow duplicates Devenv CI policy: {found!r}")


def _assert_devenv_job_recipe(job_block: str, expected_profile: str = "ci") -> None:
    """Require a converted job to use only the project Devenv toolchains."""
    required = (
        f"uses: {DEVENV_SETUP_ACTION}",
        f"uses: {DEVENV_AUTH_ACTION}",
        "app-client-id: ${{ vars.CI_APP_ID }}",
        "app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}",
        f"shell: {PORTABLE_DEVENV_SHELL}",
    )
    missing = [marker for marker in required if marker not in job_block]
    profile_command = re.search(
        rf"(?m)^\s*(?:run:\s*)?(?:if\s+)?devenv-retry --profile "
        rf"{re.escape(expected_profile)}\s",
        job_block,
    )
    if profile_command is None:
        missing.append(f"devenv-retry profile {expected_profile}")
    if missing:
        raise AssertionError(f"incomplete Devenv CI job recipe: {missing!r}")

    forbidden = (
        "dtolnay/rust-toolchain",
        "taiki-e/install-action",
        "astral-sh/setup-uv",
        "scripts/ci_setup_uv_python.py",
        ".venv/bin/python",
        ".devenv/state/venv/bin/python",
        "uv pip install",
        "repositories: ci",
        "access-tokens = github.com=",
    )
    found = [marker for marker in forbidden if marker in job_block]
    if found:
        raise AssertionError(
            f"nonportable toolchain setup remains in Devenv job: {found!r}"
        )


def _rust_cache_inputs(job_block: str) -> dict[str, str]:
    """Return the `with:` inputs for a job's Swatinem/rust-cache step."""
    lines = job_block.splitlines()
    uses_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip() == "uses: Swatinem/rust-cache@v2":
            uses_idx = idx
            break
    if uses_idx is None:
        raise AssertionError("missing Swatinem/rust-cache@v2 step")

    with_idx: int | None = None
    for idx in range(uses_idx + 1, len(lines)):
        if re.match(r"^\s*- ", lines[idx]):
            break
        if lines[idx].strip() == "with:":
            with_idx = idx
            break
    if with_idx is None:
        return {}

    inputs: dict[str, str] = {}
    for line in lines[with_idx + 1 :]:
        if re.match(r"^\s*- ", line):
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.+?)\s*$", stripped)
        if m is not None:
            inputs[m.group(1)] = m.group(2)
    return inputs


class DevenvWorkflowJobTests(unittest.TestCase):
    def test_converted_jobs_use_the_project_devenv_toolchains(self) -> None:
        focused_profiles = {
            ("ci.yml", "backend-sanitizers"): "sanitizers",
            ("ci.yml", "smt-build"): "smt",
            ("ci.yml", "smt-build-darwin-arm64"): "smt",
            ("smt-full-prove.yml", "full-smt-prove"): "smt",
        }
        for filename, jobs in PROJECT_DEVENV_WORKFLOW_JOBS.items():
            path = WORKFLOWS_DIR / filename
            for job in jobs:
                with self.subTest(filename=filename, job=job):
                    _assert_devenv_job_recipe(
                        _workflow_job_block(path, job),
                        focused_profiles.get((filename, job), "ci"),
                    )

    def test_a_wrong_focused_profile_fails_the_recipe(self) -> None:
        block = _ci_job_block("backend-sanitizers")
        mutated = block.replace("--profile sanitizers", "--profile ci", 1)
        with self.assertRaisesRegex(AssertionError, "profile sanitizers"):
            _assert_devenv_job_recipe(mutated, "sanitizers")

    def test_ci_environment_policy_is_not_duplicated_in_workflows(self) -> None:
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            with self.subTest(workflow=path.name):
                _assert_no_workflow_ci_policy(path.read_text(encoding="utf-8"))

    def test_a_duplicated_ci_environment_value_fails_the_contract(self) -> None:
        workflow = CI_YML.read_text(encoding="utf-8")
        mutated = workflow + "\nenv:\n  CARGO_PROFILE_DEV_DEBUG: 0\n"
        with self.assertRaisesRegex(AssertionError, "duplicates Devenv CI policy"):
            _assert_no_workflow_ci_policy(mutated)

    def test_host_uv_setup_fails_the_devenv_recipe(self) -> None:
        block = _ci_job_block("lint-and-unit")
        mutated = block.replace(
            "      - name: Cache cargo registry and build",
            "      - uses: astral-sh/setup-uv@v8.1.0\n\n"
            "      - name: Cache cargo registry and build",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "nonportable toolchain setup"):
            _assert_devenv_job_recipe(mutated)

    def test_direct_devenv_python_path_fails_the_recipe(self) -> None:
        block = _ci_job_block("lint-and-unit")
        mutated = block.replace(
            f"{DEVENV_COMMAND_PREFIX}python -m unittest discover",
            ".devenv/state/venv/bin/python -m unittest discover",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "nonportable toolchain setup"):
            _assert_devenv_job_recipe(mutated)


class CiParityTests(unittest.TestCase):
    """The lock: every cargo/chelis gate invocation in the CI workflow
    must be produced by `gate.py`. If a future edit hand-inlines a
    cargo command into the `lint-and-unit` or `workspace-tests` job, this
    test fails."""

    def test_ci_file_exists(self):
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")

    def test_gate_jobs_call_gate_py(self):
        # The `lint-and-unit` and `workspace-tests` jobs must invoke
        # `python3 scripts/gate.py <stage>` and must NOT hand-inline
        # any `cargo` or `chelis` command.
        invocations = _parse_ci_gate_invocations()
        for job in ("lint-and-unit", "workspace-tests"):
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

    def test_python_binding_ingress_suite_is_continuous(self):
        text = CI_YML.read_text()
        self.assertIn(
            f"{DEVENV_COMMAND_PREFIX}python -m unittest discover "
            "-s bindings/python/tests -p 'test_*.py'",
            text,
            (
                "bindings/python/tests contains the #729 Python-ingress oracle; "
                "the lint-and-unit job must discover it continuously"
            ),
        )

    def test_dtype_phase3_oracle_runs_beside_the_workspace_suite(self):
        workspace_block = _ci_job_block("workspace-tests")
        oracle_block = _ci_job_block("dtype-phase3-oracle")
        aggregate_block = _ci_job_block("integration")
        workflow = CI_YML.read_text()
        top_level_permissions = workflow[
            workflow.index("permissions:\n") : workflow.index("\nenv:\n")
        ]
        oracle_command = (
            f"run: {DEVENV_COMMAND_PREFIX}python scripts/dtype_phase3_oracle.py"
        )
        self.assertNotIn("  issues: read", top_level_permissions)
        self.assertNotIn(oracle_command, workspace_block)
        self.assertEqual(oracle_block.count("    contents: read"), 1)
        self.assertEqual(oracle_block.count("    issues: read"), 1)
        self.assertNotIn("uv pip install", oracle_block)
        self.assertIn("GH_TOKEN: ${{ github.token }}", oracle_block)
        _assert_executable_run_once(
            oracle_block, "python scripts/dtype_phase3_oracle.py"
        )
        self.assertIn("needs: [changes]", workspace_block)
        self.assertIn("needs: [changes]", oracle_block)
        self.assertEqual(oracle_block.count("    needs:"), 1)
        self.assertNotIn("needs.workspace-tests", oracle_block)
        self.assertNotIn("dtype-phase3-oracle", workspace_block)
        self.assertIn("name: Integration Tests (Linux)", aggregate_block)
        self.assertIn(
            "needs: [changes, workspace-tests, dtype-phase3-oracle, "
            "faithful-observation-phase2-oracle]",
            aggregate_block,
        )
        self.assertNotIn("always()", aggregate_block)
        self.assertIn("!cancelled()", aggregate_block)
        self.assertIn("scripts/ci_require_success.py", aggregate_block)

    def test_faithful_observation_phase2_oracle_is_a_dedicated_blocking_job(self):
        workspace_block = _ci_job_block("workspace-tests")
        dtype_block = _ci_job_block("dtype-phase3-oracle")
        oracle_block = _ci_job_block("faithful-observation-phase2-oracle")
        aggregate_block = _ci_job_block("integration")
        command = "python scripts/faithful_observation_phase2_oracle.py"

        self.assertIn("name: Faithful Observation Phase 2 Oracle", oracle_block)
        self.assertIn("needs: [changes]", oracle_block)
        self.assertIn("contents: read", oracle_block)
        self.assertIn(f"uses: {DEVENV_SETUP_ACTION}", oracle_block)
        self.assertNotIn("dtolnay/rust-toolchain@stable", oracle_block)
        self.assertNotIn("scripts/ci_setup_uv_python.py", oracle_block)
        self.assertNotIn("taiki-e/install-action@nextest", oracle_block)
        cache_inputs = _rust_cache_inputs(oracle_block)
        self.assertEqual(cache_inputs.get("shared-key"), "linux-workspace")
        self.assertEqual(cache_inputs.get("save-if"), "false")
        self.assertIn(
            "CARGO_TARGET_DIR: ${{ github.workspace }}/target",
            oracle_block,
        )
        _assert_executable_run_once(oracle_block, command)
        self.assertNotIn(command, workspace_block)
        self.assertNotIn(command, dtype_block)
        self.assertIn(
            "faithful-observation-phase2-oracle=${{ needs.faithful-observation-phase2-oracle.result }}",
            aggregate_block,
        )

    def test_parallel_jobs_share_one_saved_rust_cache_namespace(self):
        workspace_inputs = _rust_cache_inputs(_ci_job_block("workspace-tests"))
        oracle_inputs = _rust_cache_inputs(_ci_job_block("dtype-phase3-oracle"))
        self.assertEqual(workspace_inputs.get("shared-key"), "linux-workspace")
        self.assertEqual(oracle_inputs.get("shared-key"), "linux-workspace")
        self.assertNotEqual(workspace_inputs.get("save-if"), "false")
        self.assertEqual(oracle_inputs.get("save-if"), "false")

    def test_profile_partition_set_math_runs_continuously(self):
        lint_block = _ci_job_block("lint-and-unit")
        workspace_block = _ci_job_block("workspace-tests")
        self.assertIn(
            'CHELIS_SKIP_NEXTEST_PROFILE_SET_MATH: "1"',
            lint_block,
        )
        _assert_executable_run_once(
            workspace_block,
            "python -m unittest "
            "scripts.test_nextest_profile_partition.ProfilePartitionTests",
        )

    def test_quoted_oracle_name_is_not_an_executable_oracle_step(self):
        block = _ci_job_block("dtype-phase3-oracle")
        oracle_command = (
            f"run: {DEVENV_COMMAND_PREFIX}python scripts/dtype_phase3_oracle.py"
        )
        mutated = block.replace(
            oracle_command,
            'run: "true # scripts/dtype_phase3_oracle.py"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(
                mutated,
                "python scripts/dtype_phase3_oracle.py",
            )

    def test_commented_oracle_plus_noop_is_not_an_executable_oracle_step(self):
        block = _ci_job_block("dtype-phase3-oracle")
        command = "python scripts/dtype_phase3_oracle.py"
        wrapped = f"{DEVENV_COMMAND_PREFIX}{command}"
        mutated = block.replace(
            f"run: {wrapped}",
            f'# run: {wrapped}\n        run: "true"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(mutated, command)

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

    def test_all_workflow_files_are_scope_classified(self):
        # Every workflow file under .github/workflows/ must be explicitly
        # classified as out-of-scope-by-design (NON_GATE_WORKFLOWS) so adding a
        # new workflow forces a deliberate scope decision rather than silently
        # introducing commands the per-PR gate does not own.
        on_disk = {p.name for p in WORKFLOWS_DIR.glob("*.yml")}
        on_disk |= {p.name for p in WORKFLOWS_DIR.glob("*.yaml")}
        unclassified = on_disk - NON_GATE_WORKFLOWS
        self.assertEqual(
            unclassified,
            set(),
            (
                f"workflow file(s) {unclassified} are not classified in "
                f"NON_GATE_WORKFLOWS; decide explicitly whether each is part of "
                f"the per-PR gate scope"
            ),
        )

    def test_conformance_workflows_present_and_out_of_scope(self):
        # The Hull conformance gate lives in its own workflow files, out of the
        # gate.py quartet scope by design (rule-id GATE-SCOPE-CONFORMANCE).
        for name in ("conformance.yml", "conformance-nightly.yml"):
            self.assertTrue(
                (WORKFLOWS_DIR / name).is_file(), f"missing workflow {name}"
            )
            self.assertIn(name, NON_GATE_WORKFLOWS)

    def test_no_multiline_run_in_gate_jobs(self):
        # A `run: |` block in a gate job would hide its commands from
        # the line-based parity parser. Disallow it for the two gate
        # jobs so parity stays enforceable.
        invocations = _parse_ci_gate_invocations()
        for job in ("lint-and-unit", "workspace-tests"):
            self.assertNotIn(
                "<multiline-run-block>",
                invocations.get(job, []),
                (
                    f"CI gate job '{job}' uses a multi-line run block; "
                    f"keep gate steps as single-line `run: python3 "
                    f"scripts/gate.py ...` so the parity parser sees them"
                ),
            )


class NixPackagesWorkflowTests(unittest.TestCase):
    """Lock the two native Nix package jobs and their complete check command."""

    def test_native_nix_workflow_has_both_authoritative_jobs(self):
        self.assertTrue(NIX_PACKAGES_YML.is_file(), "missing Nix package workflow")
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        required = [
            "name: Nix Packages (x86_64-linux)",
            "runs-on: ubuntu-latest",
            "name: Nix Packages (aarch64-darwin)",
            "runs-on: macos-latest",
        ]
        for marker in required:
            self.assertIn(marker, text)

    def test_each_native_job_runs_the_complete_flake_check_set(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        self.assertEqual(
            text.count("run: nix flake check --print-build-logs"),
            2,
            "each native Nix job must run the complete flake check set",
        )
        self.assertNotIn(
            "scripts/gate.py",
            text,
            "Nix package jobs must stay separate from the canonical Cargo gate",
        )

    def test_each_native_job_rejects_the_wrong_runner_system(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        self.assertEqual(text.count("name: Verify the runner system"), 2)
        self.assertIn('assert system == "x86_64-linux", system', text)
        self.assertIn('assert system == "aarch64-darwin", system', text)

    def test_supported_systems_have_exact_native_job_parity(self):
        contracts = (REPO_ROOT / "nix" / "contracts.nix").read_text(encoding="utf-8")
        workflow = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_nix_system_job_parity(contracts, workflow)

    def test_supported_system_without_native_job_fails_parity(self):
        contracts = 'supportedSystems = [ "x86_64-linux" "aarch64-linux" ];'
        workflow = "name: Nix Packages (x86_64-linux)"
        with self.assertRaisesRegex(AssertionError, "aarch64-linux"):
            _assert_nix_system_job_parity(contracts, workflow)

    def test_each_native_job_runs_the_nix_contract_suite_with_devenv_python(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        self.assertNotIn("astral-sh/setup-uv", text)
        self.assertNotIn("uv venv", text)
        self.assertNotIn(".venv/bin/python", text)
        self.assertEqual(
            text.count(
                "run: devenv-retry --profile ci shell --no-tui -- "
                "python scripts/test_nix_flake_contract.py"
            ),
            2,
        )

    def test_host_python_environment_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "run: devenv-retry --profile ci shell --no-tui -- "
            "python scripts/test_nix_flake_contract.py",
            "uses: astral-sh/setup-uv@v8.1.0\n"
            "      - run: uv venv --python 3.11 .venv\n"
            "      - run: .venv/bin/python scripts/test_nix_flake_contract.py",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "host Python"):
            _assert_native_devenv_recipe(mutated)

    def test_each_native_job_uses_the_reviewed_portable_devenv_base(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_native_devenv_recipe(text)

    def test_missing_devenv_package_build_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "run: devenv-retry build --no-tui outputs.chelis "
            "outputs.chelis-runtime outputs.chelisup",
            "run: omitted",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "devenv-retry build"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_private_ci_authentication_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(f"uses: {DEVENV_AUTH_ACTION}", "uses: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "authenticate-private-ci-input"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_central_devenv_action_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(f"uses: {DEVENV_SETUP_ACTION}", "uses: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "setup-devenv"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_portable_shell_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(f"shell: {PORTABLE_DEVENV_SHELL}", "shell: bash", 1)
        with self.assertRaisesRegex(AssertionError, "devenv-ci"):
            _assert_native_devenv_recipe(mutated)

    def test_late_central_devenv_action_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        setup = f"uses: {DEVENV_SETUP_ACTION}"
        mutated = text.replace(setup, "uses: omitted", 1).replace(
            "run: devenv-retry test --no-tui",
            f"run: devenv-retry test --no-tui\n      - {setup}",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "precede the runner verification"):
            _assert_native_devenv_recipe(mutated)

    def test_linux_job_skips_docs_only_pull_requests(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_nix_docs_only_gate(text)

    def test_missing_docs_only_gate_fails_the_skip_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(f"    {DOCS_ONLY_GATE_IF}\n", "", 1)
        with self.assertRaisesRegex(AssertionError, "shared job-level gate"):
            _assert_nix_docs_only_gate(mutated)

    def test_missing_docs_only_detector_fails_the_skip_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "uses: ./.github/actions/detect-docs-only", "uses: omitted", 1
        )
        with self.assertRaisesRegex(AssertionError, "shared detector"):
            _assert_nix_docs_only_gate(mutated)

    def test_docs_only_gate_on_the_darwin_job_fails_the_skip_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "    if: github.event_name == 'workflow_dispatch'\n",
            f"    needs: [changes]\n    {DOCS_ONLY_GATE_IF}\n",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "manual dispatch as its only"):
            _assert_nix_docs_only_gate(mutated)

    def test_darwin_job_runs_on_manual_dispatch_only(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_darwin_manual_dispatch(text)

    def test_darwin_pull_request_trigger_fails_the_manual_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "    if: github.event_name == 'workflow_dispatch'\n", "", 1
        )
        with self.assertRaisesRegex(AssertionError, "manual dispatch"):
            _assert_darwin_manual_dispatch(mutated)

    def test_each_job_bounds_runner_resources(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_runner_resource_bounds(text)

    def test_unbounded_build_parallelism_fails_the_resource_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace("        max-jobs = 2\n", "", 1)
        with self.assertRaisesRegex(AssertionError, "max-jobs = 2"):
            _assert_runner_resource_bounds(mutated)

    def test_missing_disk_reclaim_fails_the_resource_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "Chelis-Lang/ci/actions/reclaim-ubuntu-runner-disk@"
            "111b5865ccf04146344ad99dcdea9d724d65fc69",
            "missing-reclaim-action",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "reclaim runner disk"):
            _assert_runner_resource_bounds(mutated)

    def test_each_job_caches_the_cvc5_toolchain_closure(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_cvc5_closure_cache(text)

    def test_missing_cvc5_restore_fails_the_cache_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "uses: ./.github/actions/cvc5-cache-restore", "uses: omitted", 1
        )
        with self.assertRaisesRegex(AssertionError, "cvc5 closure"):
            _assert_cvc5_closure_cache(mutated)

    def test_direct_devenv_bootstrap_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        setup = f"uses: {DEVENV_SETUP_ACTION}"
        duplicated = (
            f"{setup}\n"
            "      - uses: cachix/cachix-action@v17\n"
            "        with:\n"
            "          name: devenv\n"
            "          skipPush: true"
        )
        mutated = text.replace(setup, duplicated, 1)
        with self.assertRaisesRegex(AssertionError, "duplicates central setup"):
            _assert_native_devenv_recipe(mutated)


class SmtCiSplitTests(unittest.TestCase):
    """Lock the required-fast / full-prove split for SMT CI."""

    def test_required_smt_job_stays_fast_smoke(self):
        block = _ci_job_block("smt-build")
        self.assertIn("name: SMT Feature Build (Linux)", block)
        self.assertIn("verify_release_smt.py ./target/debug/chelis", block)
        self.assertIn(
            "cargo test -p chelis-prove --features smt --lib cvc5_engine_",
            block,
        )
        forbidden = [
            "--features carcara",
            "--features z3",
            '--features "smt z3"',
            "--features clarabel",
            '--features "smt clarabel"',
            "--features arb",
            "generate_erf_proof.py --check-only",
            "certify_erf_envelope",
            "Run arb-gated chelis-prove tests",
            "libz3-dev",
            "gappa",
        ]
        for needle in forbidden:
            self.assertNotIn(
                needle,
                block,
                f"required SMT smoke job must not include full-prove work: {needle}",
            )

    def test_full_smt_workflow_carries_full_prove_corpus(self):
        self.assertTrue(SMT_FULL_PROVE_YML.is_file(), "missing SMT full workflow")
        text = SMT_FULL_PROVE_YML.read_text()
        required = [
            # Nightly + manual only (the PR trigger was removed: the ~46m
            # corpus is too heavy for the per-PR path and is not a required
            # check; the per-PR cvc5 signal is ci.yml's fast smoke).
            "schedule:",
            "workflow_dispatch:",
            "shared-key: smt-smt-build",
            "cargo test -p chelis-prove --features smt",
            "chelis-z3-test --cargo-subcommand test -p chelis-prove --features z3",
            'chelis-z3-test --cargo-subcommand test -p chelis-prove --features "smt z3" --test cross_engine_oracle',
            "cargo test -p chelis-prove --features clarabel",
            'cargo test -p chelis-prove --features "smt clarabel"',
            "scripts/generate_erf_proof.py --check-only",
            "certify_erf_envelope",
            "cargo test -p chelis-prove --features arb",
            "Nightly failing: SMT Full Prove",
            "github.event_name != 'pull_request'",
            "timeout-minutes: 75",
        ]
        for needle in required:
            self.assertIn(
                needle,
                text,
                f"SMT full workflow missing expected full-prove surface: {needle}",
            )
        _assert_carcara_full_suite_command(text)
        # Negative lock: the heavy corpus must NOT run on PRs. The report job's
        # `github.event_name != 'pull_request'` guard uses a quote, not a colon,
        # so this only trips on a reintroduced `pull_request:` trigger key.
        self.assertNotIn(
            "pull_request:",
            text,
            "SMT full-prove must stay nightly/dispatch-only (no pull_request trigger)",
        )
        filtered = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"# {CARCARA_FULL_SUITE_COMMAND}\n"
            "        run: cargo test -p chelis-prove --features carcara "
            "run_carcara_check -- --test-threads=1",
        )
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(filtered)

        duplicated = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
            "      - name: Accidental parallel Carcara rerun\n"
            "        run: cargo test -p chelis-prove --features carcara",
        )
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(duplicated)

        for alternate in (
            "cargo test -p chelis-prove --features carcara,smt",
            "cargo test -p chelis-prove --features=carcara,smt",
            "cargo test -p chelis-prove -F carcara",
            "cargo test -p chelis-prove -Fcarcara",
            'cargo test -p chelis-prove --features "$FEATURES"',
            "cargo test -p chelis-prove --all-features",
        ):
            with self.subTest(alternate=alternate):
                mutated = text.replace(
                    f"run: {CARCARA_FULL_SUITE_COMMAND}",
                    f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
                    "      - name: Alternate Carcara rerun\n"
                    f"        run: {alternate}",
                )
                with self.assertRaisesRegex(
                    AssertionError, "complete serialized Carcara suite"
                ):
                    _assert_carcara_full_suite_command(mutated)

        multiline = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
            "      - name: Multiline Carcara rerun\n"
            "        run: |\n"
            "          cargo test -p chelis-prove --features carcara\n",
        )
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(multiline)

        disabled = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            "if: false\n"
            f"        run: {CARCARA_FULL_SUITE_COMMAND}\n"
            "      - name: Filtered multiline Carcara run\n"
            "        run: |\n"
            "          cargo test -p chelis-prove --features carcara "
            "run_carcara_check -- --test-threads=1\n",
        )
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(disabled)

        conditional = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"if: ${{{{ always() }}}}\n        run: {CARCARA_FULL_SUITE_COMMAND}",
        )
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(conditional)

    def test_carcara_dependency_stays_gmp_only(self):
        text = CHELIS_PROVE_TOML.read_text()
        dependency = next(
            line for line in text.splitlines() if line.startswith("gmp-mpfr-sys = ")
        )
        self.assertIn("default-features = false", dependency)
        self.assertIn("optional = true", dependency)
        self.assertNotIn(", features =", dependency)
        self.assertNotIn("gmp-mpfr-sys/mpfr", text)
        self.assertNotIn("gmp-mpfr-sys/mpc", text)
        result = subprocess.run(
            [
                "cargo",
                "tree",
                "-p",
                "chelis-prove",
                "--features",
                "carcara",
                "-e",
                "features",
                "--prefix",
                "none",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _assert_carcara_feature_tree_is_gmp_only(result.stdout)
        with self.assertRaisesRegex(AssertionError, "must stay GMP-only"):
            _assert_carcara_feature_tree_is_gmp_only(
                result.stdout + '\ngmp-mpfr-sys feature "mpfr"\nrug feature "float"'
            )

    def test_full_smt_workflow_shares_smoke_cache_key(self):
        smoke_inputs = _rust_cache_inputs(_ci_job_block("smt-build"))
        full_inputs = _rust_cache_inputs(
            _workflow_job_block(SMT_FULL_PROVE_YML, "full-smt-prove")
        )
        self.assertEqual(
            smoke_inputs,
            full_inputs,
            "required smt smoke and full-prove lane must share rust-cache inputs",
        )
        # cache-on-failure persists a warm workspace cache even when a later
        # step fails, so a slow cold run is not re-paid next attempt (#583).
        self.assertEqual(
            smoke_inputs,
            {"shared-key": "smt-smt-build", "cache-on-failure": "true"},
        )


def _parse_job_attrs() -> dict[str, dict[str, str]]:
    """Parse `.github/workflows/ci.yml` and return, per job, its
    top-level `needs:` and `if:` lines (the first occurrence at the
    job's own indent). Line-based to match the existing parser style and
    avoid a PyYAML dependency the CI venv may not carry."""
    text = CI_YML.read_text()
    lines = text.splitlines()
    current_job: str | None = None
    attrs: dict[str, dict[str, str]] = {}
    job_header = re.compile(r"^  ([a-z0-9-]+):\s*$")
    attr_line = re.compile(r"^    (needs|if):\s*(.+?)\s*$")
    # Only parse headers inside the `jobs:` block; `on:` triggers like
    # `  push:` share the two-space indent and would otherwise read as
    # jobs.
    in_jobs = False
    for line in lines:
        if line.rstrip() == "jobs:":
            in_jobs = True
            continue
        if not in_jobs:
            continue
        m = job_header.match(line)
        if m is not None:
            current_job = m.group(1)
            attrs.setdefault(current_job, {})
            continue
        if current_job is None:
            continue
        am = attr_line.match(line)
        if am is not None:
            key, value = am.group(1), am.group(2)
            attrs[current_job].setdefault(key, value)
    return attrs


def _backend_sanitizer_optimization_errors(text: str) -> list[str]:
    match = re.search(r'CHELIS_C_TEST_EXTRA_FLAGS\s*=\s*"(?P<flags>[^"]+)";', text)
    flags = "" if match is None else match.group("flags")
    if re.search(r"(?:^|\s)-O(?:1|2)(?:\s|$)", flags) is None:
        return ["backend-sanitizers: C sanitizer flags need -O1 or -O2"]
    return []


def _backend_span_compile_optimization_errors(text: str) -> list[str]:
    marker = "fn compile_kernel_only"
    start = text.find(marker)
    end = text.find("\n}\n", start)
    block = text[start:end] if start >= 0 and end >= 0 else ""
    if re.search(r'"-O(?:1|2)"', block) is None or '"-Werror"' not in block:
        return ["span_comments: compile_kernel_only needs -O1 or -O2 with -Werror"]
    return []


class BackendSanitizerWorkflowTests(unittest.TestCase):
    def test_c_sanitizer_flags_satisfy_nix_fortify(self) -> None:
        self.assertEqual(
            _backend_sanitizer_optimization_errors(DEVENV_TOOLCHAINS_NIX.read_text()),
            [],
        )

    def test_c_sanitizer_guard_rejects_missing_optimization(self) -> None:
        text = DEVENV_TOOLCHAINS_NIX.read_text()
        mutated = text.replace("-O1 ", "", 1)
        self.assertEqual(
            _backend_sanitizer_optimization_errors(mutated),
            ["backend-sanitizers: C sanitizer flags need -O1 or -O2"],
        )

    def test_span_comment_compile_satisfies_nix_fortify(self) -> None:
        self.assertEqual(
            _backend_span_compile_optimization_errors(SPAN_COMMENTS_RS.read_text()), []
        )

    def test_span_comment_guard_rejects_unoptimized_compile(self) -> None:
        text = SPAN_COMMENTS_RS.read_text()
        mutated = text.replace('"-O1"', '"-O0"', 1)
        self.assertEqual(
            _backend_span_compile_optimization_errors(mutated),
            ["span_comments: compile_kernel_only needs -O1 or -O2 with -Werror"],
        )


class MacosManualOnlyTests(unittest.TestCase):
    def test_every_macos_job_requires_manual_dispatch(self) -> None:
        found: list[tuple[str, str]] = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(workflow=path.name):
                self.assertEqual(_macos_manual_dispatch_errors(text), [])
            for job, block in _workflow_job_blocks(text).items():
                direct_macos = "runs-on: macos-latest" in block
                matrix_macos = (
                    "runs-on: ${{ matrix.os }}" in block and "os: macos-latest" in block
                )
                if direct_macos or matrix_macos:
                    found.append((path.name, job))
        self.assertEqual(
            found,
            [
                ("ci.yml", "smt-build-darwin-arm64"),
                ("ci.yml", "macos-smoke"),
                ("nix-packages.yml", "nix-darwin-arm64"),
                ("release.yml", "build-chelis-release"),
                ("release.yml", "consume-chelis-release-darwin"),
                ("release.yml", "build-chelisup-release"),
            ],
        )

    def test_macos_manual_guard_rejects_an_automatic_job(self) -> None:
        text = CI_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "    if: github.event_name == 'workflow_dispatch'",
            "    if: always()",
            1,
        )
        self.assertIn(
            "smt-build-darwin-arm64: missing manual-dispatch job condition",
            _macos_manual_dispatch_errors(mutated),
        )

    def test_macos_manual_guard_rejects_a_missing_trigger(self) -> None:
        text = CI_YML.read_text(encoding="utf-8")
        mutated = text.replace("  workflow_dispatch:\n", "", 1)
        self.assertIn(
            "macos-smoke: missing workflow_dispatch trigger",
            _macos_manual_dispatch_errors(mutated),
        )

    def test_release_runs_only_by_manual_dispatch(self) -> None:
        text = RELEASE_YML.read_text(encoding="utf-8")
        trigger = text[: text.index("\npermissions:\n")]
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotIn("push:", trigger)
        publish = _workflow_job_blocks(text)["publish-release"]
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", publish)
        self.assertNotIn("github.event_name == 'push'", publish)


class DocsOnlySkipTests(unittest.TestCase):
    """chelis#419: heavy jobs skip on docs-only PRs via a JOB-LEVEL `if`
    keyed on the `changes` job output, never `paths-ignore` (a path-
    filtered required check hangs pending forever -> merge deadlock). A
    skipped required job reports its context as success, so the skip
    direction is safe; the `if` must also fail SAFE (run the heavy job)
    when the `changes` job did not succeed, or a broken detector would
    silently skip the gate on a code PR."""

    # Jobs that must skip on a docs-only PR.
    HEAVY_GATED_JOBS = {
        "workspace-tests",
        "dtype-phase3-oracle",
        "faithful-observation-phase2-oracle",
        "backend-sanitizers",
        "smt-build",
    }
    MANUAL_ONLY_JOBS = {"macos-smoke", "smt-build-darwin-arm64"}
    # The stable required context aggregates the two parallel integration
    # legs, so it needs their results as well as the docs-only classification.
    HEAVY_AGGREGATOR_JOBS = {"integration"}
    # Jobs that use the same always-present `changes` job but key on a
    # narrower contract input rather than on the docs-only classification.
    CHANGE_GATED_JOBS = {
        "rejection-authority-liveness",
        "diagnostic-kind-oracle",
    }
    # Jobs that must ALWAYS run (never gated on docs_only).
    ALWAYS_RUN_JOBS = {
        "lint-and-unit",
        "no-ai-authorship",
        "docs",
        "changes",
    }

    def test_changes_job_exists_and_is_ungated(self):
        attrs = _parse_job_attrs()
        self.assertIn("changes", attrs, "the docs-only detector job must exist")
        # The changes job itself must not be gated on its own output and
        # must always run so its result/output are well-defined.
        self.assertNotIn("if", attrs["changes"])
        self.assertNotIn("needs", attrs["changes"])

    def test_changes_jobs_consume_the_shared_detector_action(self):
        # The diff-range logic lives once, in the local composite action;
        # every workflow with a changes job consumes that single copy, and
        # the action itself pipes the diff through the shared Python
        # detector. A checkout must precede the local action reference.
        action = (
            WORKFLOWS_DIR.parent / "actions" / "detect-docs-only" / "action.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/ci_detect_docs_only.py", action)
        self.assertIn("git diff --name-only", action)
        for path in (CI_YML, NIX_PACKAGES_YML, WORKFLOWS_DIR / "conformance.yml"):
            with self.subTest(workflow=path.name):
                block = _workflow_job_block(path, "changes")
                self.assertIn("uses: ./.github/actions/detect-docs-only", block)
                self.assertLess(
                    block.index("uses: actions/checkout@"),
                    block.index("uses: ./.github/actions/detect-docs-only"),
                    "the local action needs the repository checked out first",
                )

    def test_no_paths_ignore_in_workflow(self):
        # paths-ignore / paths on a required check deadlocks branch
        # protection; the whole point of #419 is to use job-level `if`
        # instead. Match real YAML keys, not the explanatory comment that
        # names `paths-ignore` to warn against it.
        for raw in CI_YML.read_text().splitlines():
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            self.assertFalse(
                stripped.startswith("paths-ignore:") or stripped.startswith("paths:"),
                f"ci.yml uses a path filter ({stripped!r}); a path-filtered "
                f"required check hangs pending forever. Use a job-level `if` "
                f"on the changes output instead (chelis#419)",
            )

    def test_heavy_jobs_gate_on_changes_failsafe(self):
        attrs = _parse_job_attrs()
        for job in self.HEAVY_GATED_JOBS:
            self.assertIn(job, attrs, f"heavy job '{job}' missing")
            self.assertEqual(
                attrs[job].get("needs"),
                "[changes]",
                f"'{job}' must `needs: [changes]` to read docs_only",
            )
            cond = attrs[job].get("if", "")
            # Must reference the docs_only output ...
            self.assertIn(
                "needs.changes.outputs.docs_only != 'true'",
                cond,
                f"'{job}' if must skip only when docs_only == 'true': {cond!r}",
            )
            # ... fail SAFE when the changes job did not succeed ...
            self.assertIn(
                "needs.changes.result != 'success'",
                cond,
                f"'{job}' if must run when the changes job failed: {cond!r}",
            )
            # ... and not run on cancellation.
            self.assertIn(
                "!cancelled()",
                cond,
                f"'{job}' if must include !cancelled(): {cond!r}",
            )

    def test_macos_jobs_run_only_by_manual_dispatch(self):
        attrs = _parse_job_attrs()
        for job in self.MANUAL_ONLY_JOBS:
            self.assertIn(job, attrs, f"manual job '{job}' missing")
            self.assertNotIn("needs", attrs[job])
            self.assertEqual(
                attrs[job].get("if"),
                "github.event_name == 'workflow_dispatch'",
            )

    def test_integration_aggregator_is_fail_closed_and_docs_gated(self):
        attrs = _parse_job_attrs()
        integration = attrs["integration"]
        self.assertEqual(
            integration.get("needs"),
            "[changes, workspace-tests, dtype-phase3-oracle, "
            "faithful-observation-phase2-oracle]",
        )
        cond = integration.get("if", "")
        self.assertNotIn("always()", cond)
        self.assertIn("!cancelled()", cond)
        self.assertIn("needs.changes.result != 'success'", cond)
        self.assertIn("needs.changes.outputs.docs_only != 'true'", cond)
        block = _ci_job_block("integration")
        self.assertIn("needs.workspace-tests.result", block)
        self.assertIn("needs.dtype-phase3-oracle.result", block)
        self.assertIn("needs.faithful-observation-phase2-oracle.result", block)
        self.assertIn("scripts/ci_require_success.py", block)

    def test_always_run_jobs_are_not_gated(self):
        attrs = _parse_job_attrs()
        for job in self.ALWAYS_RUN_JOBS:
            self.assertIn(job, attrs, f"always-run job '{job}' missing")
            self.assertNotIn(
                "needs",
                attrs[job],
                f"always-run job '{job}' must not gate on changes",
            )
            self.assertNotIn(
                "if",
                attrs[job],
                f"always-run job '{job}' must not carry a docs_only `if`",
            )

    def test_every_job_is_classified(self):
        # Every ci.yml job is either heavy-gated or always-run; a new job
        # forces a deliberate classification (mirrors the workflow-file
        # scope test).
        attrs = _parse_job_attrs()
        classified = (
            self.HEAVY_GATED_JOBS
            | self.HEAVY_AGGREGATOR_JOBS
            | self.CHANGE_GATED_JOBS
            | self.MANUAL_ONLY_JOBS
            | self.ALWAYS_RUN_JOBS
        )
        unclassified = set(attrs) - classified
        self.assertEqual(
            unclassified,
            set(),
            f"ci.yml job(s) {unclassified} are not classified docs-only-"
            f"skip vs always-run; decide explicitly (chelis#419)",
        )


class NoAiAuthorshipTests(unittest.TestCase):
    def test_required_job_uses_the_reviewed_shared_profile(self):
        block = _ci_job_block("no-ai-authorship")
        self.assertIn("name: No AI authorship markers", block)
        self.assertIn(
            "uses: Chelis-Lang/ci/actions/check-authorship@"
            "111b5865ccf04146344ad99dcdea9d724d65fc69",
            block,
        )
        self.assertIn("profile: all-markers", block)
        self.assertNotIn("AI_MSG_PATTERNS", block)
        self.assertNotIn("AI_IDENTITY_PATTERNS", block)


class TomllibImportGuardTests(unittest.TestCase):
    """chelis#366: `tomllib` is stdlib only from Python 3.11. A top-level
    import crashed EVERY gate.py invocation under the macOS system
    `python3` (3.9), including `--list` and the per-stage CI forms that
    never parse TOML. The import is deferred into
    `workspace_member_packages()` and guarded with a guidance message."""

    def test_module_has_no_top_level_tomllib_import(self):
        # The deferred import must NOT be reintroduced at module top.
        source = (REPO_ROOT / "scripts" / "gate.py").read_text()
        lines = source.splitlines()
        in_func = False
        for line in lines:
            # Any line that starts a top-level def/class ends the import
            # region we care about; function-local `import tomllib` is fine.
            if (
                line
                and not line[0].isspace()
                and (line.startswith("def ") or line.startswith("class "))
            ):
                in_func = True
            stripped = line.strip()
            if (
                stripped == "import tomllib"
                and not in_func
                and (
                    line == stripped  # zero indentation == module top
                )
            ):
                self.fail(
                    "gate.py imports tomllib at module top; defer it into "
                    "workspace_member_packages() so --list works under "
                    "Python <3.11 (chelis#366)"
                )

    def test_list_works_without_tomllib(self):
        # Import the module and run `--list` with tomllib hidden, proving
        # neither the import nor `--list` touches tomllib. We exec the
        # module source in a namespace whose __import__ raises for tomllib.
        source = (REPO_ROOT / "scripts" / "gate.py").read_text()
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("No module named 'tomllib'")
            return real_import(name, *args, **kwargs)

        ns: dict = {
            "__name__": "gate_under_test",
            "__file__": str(REPO_ROOT / "scripts" / "gate.py"),
            "__builtins__": dict(__builtins__)
            if isinstance(__builtins__, dict)
            else dict(vars(__builtins__)),
        }
        ns["__builtins__"]["__import__"] = fake_import
        # exec must not raise: the top-level import block has no tomllib.
        exec(compile(source, "gate.py", "exec"), ns)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ns["main"](["--list"])
        self.assertEqual(rc, 0)
        self.assertIn("cargo nextest run --workspace", buf.getvalue())

    def test_workspace_member_packages_gives_guidance_without_tomllib(self):
        # The deferred import path must raise SystemExit with the guidance
        # string (not a raw ModuleNotFoundError traceback) under <3.11.
        source = (REPO_ROOT / "scripts" / "gate.py").read_text()
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tomllib":
                raise ModuleNotFoundError("No module named 'tomllib'")
            return real_import(name, *args, **kwargs)

        ns: dict = {
            "__name__": "gate_under_test",
            "__file__": str(REPO_ROOT / "scripts" / "gate.py"),
            "__builtins__": dict(__builtins__)
            if isinstance(__builtins__, dict)
            else dict(vars(__builtins__)),
        }
        ns["__builtins__"]["__import__"] = fake_import
        exec(compile(source, "gate.py", "exec"), ns)
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stderr(err):
            ns["workspace_member_packages"]()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Python 3.11+", err.getvalue())
        self.assertIn("route through uv", err.getvalue())


if __name__ == "__main__":
    unittest.main()
