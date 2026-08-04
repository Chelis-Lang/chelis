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
SMT_FULL_PROVE_YML = REPO_ROOT / ".github" / "workflows" / "smt-full-prove.yml"
NIX_PACKAGES_YML = REPO_ROOT / ".github" / "workflows" / "nix-packages.yml"
DEVENV_SETUP_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@"
    "73f017c4d3179dc313844e9d5f08d17a7879c824"
)
PORTABLE_DEVENV_SHELL = "devenv-ci bash --noprofile --norc -e -o pipefail {0}"
DOCS_ONLY_GATE_IF = (
    "if: ${{ !cancelled() && (needs.changes.result != 'success' "
    "|| needs.changes.outputs.docs_only != 'true') }}"
)
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
DEVENV_COMPOSITION_TESTS = (
    "run: .venv/bin/python -m unittest scripts.test_devenv_version "
    "scripts.test_devenv_composition scripts.test_check_nix_lock_parity"
)
DEVENV_OUTPUT_BUILD = (
    "run: devenv build --no-tui outputs.chelis outputs.chelis-runtime "
    "outputs.chelisup outputs.default > .devenv-package-outputs.json"
)
DEVENV_GRAPH_BUILD = (
    "run: devenv build --no-tui "
    "chelis.rust.workspaceGraph.generatedCargoNix "
    "> .devenv-workspace-graph.json"
)
DEVENV_OUTPUT_CHECK = (
    "run: .venv/bin/python scripts/check_devenv_package_outputs.py "
    ".devenv-package-outputs.json .devenv-workspace-graph.json"
)
DEVENV_PACKAGE_CHECKER = REPO_ROOT / "scripts" / "check_devenv_package_outputs.py"


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


def _assert_native_devenv_recipe(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    for job in ("nix-linux-x86-64", "nix-darwin-arm64"):
        if job not in blocks:
            raise AssertionError(f"missing native Nix job {job!r}")
        block = blocks[job]
        required_markers = (
            f"uses: {DEVENV_SETUP_ACTION}",
            f"shell: {PORTABLE_DEVENV_SHELL}",
            "run: devenv test --no-tui",
            DEVENV_COMPOSITION_TESTS,
            DEVENV_OUTPUT_BUILD,
            DEVENV_GRAPH_BUILD,
            DEVENV_OUTPUT_CHECK,
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

    forbidden_markers = (
        "uses: cachix/install-nix-action@",
        "uses: cachix/cachix-action@",
        "nix profile add github:cachix/devenv/",
    )
    for marker in forbidden_markers:
        if marker in workflow:
            raise AssertionError(
                f"native Devenv recipe duplicates central setup marker {marker!r}"
            )


def _assert_nix_docs_only_gate(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    changes = blocks.get("changes", "")
    if "scripts/ci_detect_docs_only.py" not in changes:
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
    reclaim_index = linux.find("name: Reclaim runner disk space")
    setup_index = linux.find(f"uses: {DEVENV_SETUP_ACTION}")
    if reclaim_index < 0 or setup_index < 0 or reclaim_index > setup_index:
        raise AssertionError(
            "the Linux Nix job must reclaim runner disk before Devenv setup"
        )


def _assert_cvc5_closure_cache(workflow: str) -> None:
    blocks = _workflow_job_blocks(workflow)
    jobs = (
        ("nix-linux-x86-64", "x86_64-linux"),
        ("nix-darwin-arm64", "aarch64-darwin"),
    )
    for job, system in jobs:
        block = blocks.get(job, "")
        markers = (
            "uses: actions/cache/restore@v4",
            "uses: actions/cache/save@v4",
            f".#legacyPackages.{system}.cvc5-dir.drvPath",
            f".#legacyPackages.{system}.cvc5-dir.outPath",
            "--no-check-sigs",
        )
        for marker in markers:
            if marker not in block:
                raise AssertionError(
                    f"{job!r} must cache the cvc5 closure: missing {marker!r}"
                )
        check_index = block.index("run: nix flake check")
        if block.index("uses: actions/cache/restore@v4") > check_index:
            raise AssertionError(
                f"{job!r} must restore the cvc5 closure before the flake check"
            )
        if block.index("uses: actions/cache/save@v4") < check_index:
            raise AssertionError(
                f"{job!r} must save the cvc5 closure after the flake check"
            )


# CI jobs that are deliberately NOT part of the per-PR developer gate.
# `gate.py` only owns the `lint-and-unit` and `integration` jobs; these
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
    # Rule-id: GATE-SCOPE-SMT -- the smt-build job is the required fast
    # cvc5-backed `smt` feature smoke. It is out of gate.py scope by
    # design, like backend-sanitizers; the full prove corpus lives in
    # smt-full-prove.yml. Runbook:
    # docs/smt_build_setup.md.
    "smt-build",
    # Rule-id: GATE-SCOPE-SMT -- the chelis#422 prove-in-CI lanes that
    # build `chelis-cli --features smt` (cvc5 from source) on the two
    # release targets release.yml ships the feature to but ubuntu's
    # smt-build does not cover: the glibc-2.31 (debian:11) compat
    # toolchain and macOS-arm64. Same from-source cvc5 cost as smt-build,
    # so out of the per-PR gate scope by design.
    "smt-build-glibc231",
    "smt-build-darwin-arm64",
}

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
    # Producer for the durable prebuilt-cvc5 Release asset the smt lanes LINK
    # (scripts/ci_cvc5_cache.py). Builds cvc5 from source and publishes a
    # Release; it runs no cargo/chelis command the per-PR gate owns, only on a
    # cvc5-sys bump / dispatch / weekly schedule. Out of gate.py scope.
    "build-cvc5.yml",
    # Native Nix package jobs build the complete flake check set on Linux and
    # macOS. These jobs prove a separate source-build channel and do not run
    # commands from the canonical Cargo gate.
    "nix-packages.yml",
    # Scheduled Actions-cache pruner (scripts/ci_cache_prune.py). Deletes stale
    # caches to hold the pool under the 10GB LRU budget; runs no per-PR gate
    # command. Out of gate.py scope by design.
    "cache-prune.yml",
    # OpenSpec validation uses the pinned central action in advisory mode.
    # It runs no cargo or Chelis command that the developer gate owns.
    # It stays outside gate.py by design.
    "openspec-validate.yml",
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
        # oracle before push rather than in CI, so it belongs in `--local`
        # too (chelis#875).
        rendered = [gate.render(c) for c in gate.LOCAL_STATIC_COMMANDS]
        self.assertIn("cargo test -p chelis-types --doc", rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_lint_and_unit_stage(self):
        rendered = [
            gate.render(command) for command in gate.STAGES["lint-and-unit"]
        ]
        for command in (
            "cargo test -p chelis-compiler-api --doc",
            ".venv/bin/python scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_local_subset(self):
        rendered = [gate.render(command) for command in gate.LOCAL_STATIC_COMMANDS]
        for command in (
            "cargo test -p chelis-compiler-api --doc",
            ".venv/bin/python scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

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


def _ci_job_block(job: str) -> str:
    """Return the raw `.github/workflows/ci.yml` text block for one job."""
    return _workflow_job_block(CI_YML, job)


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

    def test_python_binding_ingress_suite_is_continuous(self):
        text = CI_YML.read_text()
        self.assertIn(
            ".venv/bin/python -m unittest discover -s bindings/python/tests "
            "-p 'test_*.py'",
            text,
            (
                "bindings/python/tests contains the #729 Python-ingress oracle; "
                "the lint-and-unit job must discover it continuously"
            ),
        )

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

    def test_each_native_job_runs_the_nix_contract_suite_with_project_python(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        self.assertEqual(text.count("uses: astral-sh/setup-uv@v8.1.0"), 2)
        self.assertEqual(text.count("run: uv venv --python 3.11 .venv"), 2)
        self.assertEqual(
            text.count("run: .venv/bin/python scripts/test_nix_flake_contract.py"),
            2,
        )

    def test_each_native_job_uses_the_reviewed_portable_devenv_base(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_native_devenv_recipe(text)

    def test_missing_devenv_composition_tests_fail_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(DEVENV_COMPOSITION_TESTS, "run: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "test_devenv_composition"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_devenv_package_build_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(DEVENV_OUTPUT_BUILD, "run: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "devenv build"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_devenv_graph_build_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(DEVENV_GRAPH_BUILD, "run: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "workspaceGraph"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_devenv_package_check_fails_the_native_recipe(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(DEVENV_OUTPUT_CHECK, "run: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "check_devenv_package_outputs"):
            _assert_native_devenv_recipe(mutated)

    def test_devenv_package_checker_covers_native_behavior_and_smt(self):
        checker = DEVENV_PACKAGE_CHECKER.read_text(encoding="utf-8")
        required = (
            "check_inventories",
            "check_behavior",
            "Cargo-generated.nix",
            "verify_release_smt.py",
            "runtime-consumer.c",
            "shellcheck",
            "outputs.default",
        )
        for marker in required:
            self.assertIn(marker, checker)

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
            "run: devenv test --no-tui",
            f"run: devenv test --no-tui\n      - {setup}",
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
        mutated = text.replace("scripts/ci_detect_docs_only.py", "omitted.py", 1)
        with self.assertRaisesRegex(AssertionError, "shared detector"):
            _assert_nix_docs_only_gate(mutated)

    def test_docs_only_gate_on_the_darwin_job_fails_the_skip_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace(
            "    if: github.event_name == 'workflow_dispatch'\n",
            "    needs: [changes]\n"
            f"    {DOCS_ONLY_GATE_IF}\n",
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
        mutated = text.replace("name: Reclaim runner disk space", "name: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "reclaim runner disk"):
            _assert_runner_resource_bounds(mutated)

    def test_each_job_caches_the_cvc5_toolchain_closure(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        _assert_cvc5_closure_cache(text)

    def test_missing_cvc5_restore_fails_the_cache_lock(self):
        text = NIX_PACKAGES_YML.read_text(encoding="utf-8")
        mutated = text.replace("uses: actions/cache/restore@v4", "uses: omitted", 1)
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
            "cargo test -p chelis-prove --features carcara",
            "cargo test -p chelis-prove --features z3",
            'cargo test -p chelis-prove --features "smt z3" --test cross_engine_oracle',
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
        # Negative lock: the heavy corpus must NOT run on PRs. The report job's
        # `github.event_name != 'pull_request'` guard uses a quote, not a colon,
        # so this only trips on a reintroduced `pull_request:` trigger key.
        self.assertNotIn(
            "pull_request:",
            text,
            "SMT full-prove must stay nightly/dispatch-only (no pull_request trigger)",
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
        "integration",
        "macos-smoke",
        "backend-sanitizers",
        "smt-build",
    }
    # Jobs that must ALWAYS run (never gated on docs_only).
    # smt-build-glibc231 / smt-build-darwin-arm64 were added by chelis#422
    # (ship-smt) without a docs_only `if`, so today they run unconditionally
    # and are classified here. Follow-up: give them the same docs-skip `if` +
    # `needs: [changes]` as smt-build and move them to HEAVY_GATED_JOBS so the
    # heavy from-source cvc5 builds also skip on docs-only PRs (chelis#419).
    ALWAYS_RUN_JOBS = {
        "lint-and-unit",
        "no-ai-authorship",
        "docs",
        "changes",
        "smt-build-glibc231",
        "smt-build-darwin-arm64",
    }

    def test_changes_job_exists_and_is_ungated(self):
        attrs = _parse_job_attrs()
        self.assertIn(
            "changes", attrs, "the docs-only detector job must exist"
        )
        # The changes job itself must not be gated on its own output and
        # must always run so its result/output are well-defined.
        self.assertNotIn("if", attrs["changes"])
        self.assertNotIn("needs", attrs["changes"])

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
                stripped.startswith("paths-ignore:")
                or stripped.startswith("paths:"),
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
        classified = self.HEAVY_GATED_JOBS | self.ALWAYS_RUN_JOBS
        unclassified = set(attrs) - classified
        self.assertEqual(
            unclassified,
            set(),
            f"ci.yml job(s) {unclassified} are not classified docs-only-"
            f"skip vs always-run; decide explicitly (chelis#419)",
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
            if line and not line[0].isspace() and (
                line.startswith("def ") or line.startswith("class ")
            ):
                in_func = True
            stripped = line.strip()
            if stripped == "import tomllib" and not in_func and (
                line == stripped  # zero indentation == module top
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

        ns: dict = {"__name__": "gate_under_test", "__file__": str(
            REPO_ROOT / "scripts" / "gate.py"
        ), "__builtins__": dict(__builtins__) if isinstance(
            __builtins__, dict
        ) else dict(vars(__builtins__))}
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

        ns: dict = {"__name__": "gate_under_test", "__file__": str(
            REPO_ROOT / "scripts" / "gate.py"
        ), "__builtins__": dict(__builtins__) if isinstance(
            __builtins__, dict
        ) else dict(vars(__builtins__))}
        ns["__builtins__"]["__import__"] = fake_import
        exec(compile(source, "gate.py", "exec"), ns)
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stderr(err):
            ns["workspace_member_packages"]()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Python 3.11+", err.getvalue())
        self.assertIn(".venv/bin/python", err.getvalue())


if __name__ == "__main__":
    unittest.main()
