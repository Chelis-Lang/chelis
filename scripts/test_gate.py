"""Unit tests for `gate.py`.

Run via:
`uv run --managed-python --python 3.11 --no-project python -m unittest
scripts.test_gate` from the repo root.

Four things are locked here:

  (a) the full developer gate keeps the complete default nextest profile while
      CI delegates its two census binaries from the `ci` profile to the dtype
      oracle;
  (b) a structural parity assertion: every single-line `run:` scalar in a
      gate-owned job exactly matches that job's reviewed allowlist. This does
      not emulate Bash; quoting, expansion, and substitution cannot hide an
      added command. Every other CI job is classified by name so the scope
      exclusion is explicit and reviewable;
  (c) `--list` prints the canonical full list;
  (d) no-ai-authorship patterns cover the current banned tool identities.
"""

import hashlib
import importlib.util
import io
import os
import re
from collections import deque
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gate", here / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_oracle_module():
    """Load the chelis#908 oracle the same way `_load_module` loads gate.py.

    By path, so this does not depend on `sys.path` happening to carry the
    scripts directory, and under its real module name so it is the same
    object `gate.py`'s own import produced.
    """
    here = Path(__file__).resolve().parent
    name = "unrepresentable_domain_oracle"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()
REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PR_CONTRACT_ACK_YML = (
    REPO_ROOT / ".github" / "workflows" / "pr-contract-acknowledgements.yml"
)
PR_PACKAGE_EXPANSION_YML = (
    REPO_ROOT / ".github" / "workflows" / "pr-package-expansion.yml"
)
SMT_FULL_PROVE_YML = REPO_ROOT / ".github" / "workflows" / "smt-full-prove.yml"
CHELIS_PROVE_TOML = REPO_ROOT / "crates" / "chelis-prove" / "Cargo.toml"
NIX_PACKAGES_YML = REPO_ROOT / ".github" / "workflows" / "nix-packages.yml"
DEVENV_SETUP_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@"
    "73f017c4d3179dc313844e9d5f08d17a7879c824"
)
PORTABLE_DEVENV_SHELL = "devenv-ci bash --noprofile --norc -e -o pipefail {0}"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CARCARA_FULL_SUITE_COMMAND = (
    "cargo test -p chelis-prove --features carcara -- --test-threads=1"
)
SMT_FULL_SYSTEM_PACKAGES = (
    "gcc",
    "g++",
    "cmake",
    "libclang-dev",
    "m4",
    "make",
    "libz3-dev",
)
SMT_FULL_VENDORED_SYSTEM_PACKAGES = (
    "libgmp-dev",
    "libmpfr-dev",
    "libmpc-dev",
    "libopenblas-dev",
)


def _read_nix_packages_workflow(path: Path | None = None) -> str:
    """Read the exact UTF-8 workflow blob without path or newline substitution."""
    source = NIX_PACKAGES_YML if path is None else path
    if source.is_symlink() or not source.is_file():
        raise AssertionError("the Nix package workflow must be a regular file")
    raw = source.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("the Nix package workflow must be UTF-8") from error


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
    headers: list[tuple[str, int]] = []
    in_jobs = False
    offset = 0
    for line in workflow.splitlines(keepends=True):
        clean_line = line.rstrip("\r\n")
        if _parse_workflow_jobs_header(clean_line):
            in_jobs = True
        elif in_jobs:
            name = _parse_workflow_job_header(clean_line)
            if name is not None:
                headers.append((name, offset))
        offset += len(line)

    blocks: dict[str, str] = {}
    for index, (name, start) in enumerate(headers):
        end = headers[index + 1][1] if index + 1 < len(headers) else len(workflow)
        blocks[name] = workflow[start:end]
    return blocks


def _assert_carcara_full_suite_command(workflow: str) -> None:
    block = _workflow_job_blocks(workflow).get("full-smt-prove")
    if block is None:
        raise AssertionError("missing full-smt-prove job")

    def run_steps() -> list[tuple[str, bool]]:
        lines = block.splitlines()
        steps_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() == "steps:"
            ),
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

    # Cargo subcommands that only type-check or lint. The invariant here is
    # that the Carcara *suite* runs exactly once and serialized, because a
    # second concurrent run is the hazard; a step that merely compiles the
    # feature executes no test and cannot violate it. The list is of
    # non-executing subcommands rather than executing ones, so an unrecognized
    # subcommand is treated as executing and the guard stays strict.
    non_executing = frozenset(
        ("check", "clippy", "doc", "fmt", "metadata", "tree", "verify-project")
    )

    def executes_tests(command: str) -> bool:
        try:
            words = shlex.split(command, comments=True)
        except ValueError:
            return True
        for index, word in enumerate(words):
            if word != "cargo":
                continue
            for candidate in words[index + 1 :]:
                if candidate.startswith("+"):
                    continue
                return candidate not in non_executing
        return True

    steps = run_steps()
    carcara_steps = [
        (command, conditional)
        for command, conditional in steps
        if enables_carcara(command) and executes_tests(command)
    ]
    canonical_words = shlex.split(CARCARA_FULL_SUITE_COMMAND)
    canonical_steps = [
        command
        for command, conditional in carcara_steps
        if not conditional
        and shlex.split(command, comments=True) == canonical_words
    ]
    if len(carcara_steps) != 1 or canonical_steps != [CARCARA_FULL_SUITE_COMMAND]:
        raise AssertionError(
            "full-smt-prove must execute the complete serialized Carcara suite "
            f"exactly once in an unconditional step; found {carcara_steps}"
        )


def _assert_full_smt_system_packages(workflow: str) -> None:
    block = _workflow_job_blocks(workflow).get("full-smt-prove")
    if block is None:
        raise AssertionError("missing full-smt-prove job")
    commands = [
        line.strip().removeprefix("run: ")
        for line in block.splitlines()
        if line.strip().startswith("run: python3 scripts/ci_apt_get.py ")
    ]
    forbidden = sorted(
        {
            package
            for command in commands
            for package in SMT_FULL_VENDORED_SYSTEM_PACKAGES
            if package in shlex.split(command)
        }
    )
    if forbidden:
        raise AssertionError(
            "full-smt-prove must build its locked native dependencies instead "
            "of installing distribution packages: " + ", ".join(forbidden)
        )
    expected = "python3 scripts/ci_apt_get.py " + " ".join(SMT_FULL_SYSTEM_PACKAGES)
    if commands.count(expected) != 1:
        raise AssertionError(
            "full-smt-prove system dependency set must provision the headers "
            f"needed by its unified all-features build; expected {expected!r}, "
            f"found {commands!r}"
        )


def _assert_prove_uses_vendored_gmp_family(
    manifest: str, carcara_feature_tree: str, all_feature_tree: str
) -> None:
    active_manifest = "\n".join(
        line.split("#", 1)[0] for line in manifest.splitlines()
    )
    if (
        "use-system-libs" in active_manifest
        or "use-system-libs" in carcara_feature_tree
        or "use-system-libs" in all_feature_tree
    ):
        raise AssertionError(
            "chelis-prove must build the locked GMP/MPFR/MPC sources instead "
            "of forcing distribution system-library versions"
        )
    direct_dependency = re.search(
        r"^gmp-mpfr-sys\s*=", active_manifest, re.MULTILINE
    )
    if direct_dependency is not None:
        raise AssertionError(
            "chelis-prove must not retain a direct gmp-mpfr-sys dependency "
            "whose only purpose was forcing system libraries"
        )

    forbidden = (
        'gmp-mpfr-sys feature "mpfr"',
        'gmp-mpfr-sys feature "mpc"',
        'rug feature "float"',
        'rug feature "complex"',
    )
    active = [feature for feature in forbidden if feature in carcara_feature_tree]
    if active:
        raise AssertionError(
            "Carcara feature graph must stay GMP-only; activated "
            + ", ".join(active)
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
            "run: devenv test --no-tui",
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


_SIMPLE_YAML_KEY = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_-]*|'(?:[^']|'')*'|\"[^\"\\]*\")"
)


def _parse_simple_yaml_mapping_entry(
    line: str, *, expected_indent: int
) -> tuple[str, str]:
    """Parse one deliberately restricted YAML mapping entry.

    The workflow policy oracle has no YAML dependency, so it accepts the plain
    and simply quoted keys GitHub accepts and fails closed on anchors, explicit
    keys, escaped double-quoted keys, and other shapes it cannot attribute.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        raise AssertionError(f"expected YAML mapping entry: {line!r}")
    indent = len(line) - len(line.lstrip(" "))
    if indent != expected_indent:
        raise AssertionError(f"unsupported YAML indentation: {line!r}")
    match = re.fullmatch(
        rf" {{{expected_indent}}}(?P<key>{_SIMPLE_YAML_KEY.pattern})\s*:\s*(?P<value>.*)",
        line,
    )
    if match is None:
        raise AssertionError(f"unsupported YAML mapping entry: {line!r}")

    raw_key = match.group("key")
    if raw_key[0] == raw_key[-1] and raw_key[0] in {"'", '"'}:
        key = raw_key[1:-1]
        if raw_key[0] == "'":
            key = key.replace("''", "'")
    else:
        key = raw_key
    value = re.sub(r"\s+#.*$", "", match.group("value")).strip()
    return key, value


def _nix_workflow_events(workflow: str) -> dict[str, dict[str, str]]:
    lines = workflow.splitlines()
    on_index: int | None = None
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent != 0:
            continue
        entry = _parse_simple_yaml_mapping_entry(line, expected_indent=0)
        if entry[0] == "on":
            if on_index is not None:
                raise AssertionError("the Nix workflow must define one on map")
            if entry[1]:
                raise AssertionError("the Nix workflow on map must use block form")
            on_index = index
    if on_index is None:
        raise AssertionError("the Nix workflow must define an on map")

    events: dict[str, dict[str, str]] = {}
    current_event: str | None = None
    for line in lines[on_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break
        if indent == 2:
            event, value = _parse_simple_yaml_mapping_entry(
                line, expected_indent=2
            )
            if event in events:
                raise AssertionError(f"duplicate Nix workflow event {event!r}")
            if value:
                raise AssertionError(
                    f"Nix workflow event {event!r} must use block form"
                )
            events[event] = {}
            current_event = event
            continue
        if indent == 4 and current_event is not None:
            key, value = _parse_simple_yaml_mapping_entry(
                line, expected_indent=4
            )
            if key in events[current_event]:
                raise AssertionError(
                    f"duplicate {current_event!r} event key {key!r}"
                )
            events[current_event][key] = value
            continue
        raise AssertionError(f"unsupported Nix workflow event entry: {line!r}")
    return events


_NIX_REVIEWED_WORKFLOW_SHA256 = (
    "6f4f661e55a108b30d3bb2e6f463fbcde39c4f69e7b94f50f16c6b4ef721850b"
)


def _assert_nix_workflow_matches_reviewed_recipe(workflow: str) -> None:
    """Lock the workflow while the temporary event policy is active.

    YAML has enough scalar and expression spellings that a partial parser or a
    banlist can accept a semantically gated job. Bind every workflow byte so
    top-level defaults, concurrency, permissions, triggers, jobs, and steps are
    one closed reviewed artifact. Retire this digest in the reviewed change
    that ends the temporary manual-dispatch/published-release-only policy,
    after replacement event/native-recipe controls and the complete native
    flake and Devenv smoke checks pass on both supported systems under that
    policy. A recipe edit while the temporary policy stands still requires
    an explicit reviewed freeze move; it does not authorize retirement.
    """
    actual = hashlib.sha256(workflow.encode("utf-8")).hexdigest()
    if actual != _NIX_REVIEWED_WORKFLOW_SHA256:
        raise AssertionError(
            "the intentional-event-only Nix workflow must match the reviewed "
            f"native recipe and event policy: found SHA-256 {actual}"
        )


def _assert_nix_intentional_events_only(workflow: str) -> None:
    events = _nix_workflow_events(workflow)
    expected_events = {
        "workflow_dispatch": {},
        "release": {"types": "[published]"},
    }
    if events != expected_events:
        raise AssertionError(
            "the Nix workflow must run only on manual dispatch and published "
            f"releases: found {events!r}"
        )

    _assert_nix_workflow_matches_reviewed_recipe(workflow)


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


# Every single-line command allowed in the three gate-owned workers. Comparing
# complete scalars is intentionally stricter than recognizing Cargo through
# Bash syntax: an added command of any kind requires an explicit review here.
GATE_WORKER_RUN_COMMANDS = {
    "lint-rust": (
        "python3 scripts/ci_apt_get.py gcc libopenblas-dev libasan8 libubsan1",
        "python3 scripts/ci_setup_uv_python.py",
        "python3 scripts/gate.py lint-and-unit",
    ),
    "ci-fast": (
        # `clang` is the chelis#893 Phase 0 oracle's C/Objective-C front end
        # and the census's `cc` is gcc here; the image ships clang, but the
        # gate asserts it rather than assuming it.
        "python3 scripts/ci_apt_get.py gcc clang libopenblas-dev libasan8 libubsan1",
        "python3 scripts/ci_setup_uv_python.py",
        "uv pip install --python .venv/bin/python -r bindings/python/pyproject.toml",
        "python3 scripts/gate.py ci-fast",
    ),
    # Rule-id: GATE-STAGE-RUNTIME-REPRESENTATION -- chelis#893 Phase 0's oracle
    # is its own gate stage because its release-profile reproducers and serial
    # mutation re-scans cost about eleven hosted minutes, which doubled the
    # workspace shard that used to carry it. The job restores the read-only
    # workspace cache and needs `clang` for the C/Objective-C header lanes.

}


# CI jobs that are deliberately NOT part of the per-PR developer gate.
# `gate.py` only owns the three workers above; every other job is listed by name
# so a new job cannot silently escape a scope decision.
NON_GATE_JOBS = {
    # CI-owned Python unit coverage; it runs no canonical gate stage.
    "script-unit",
    # chelis#1824 derives and executes package-qualified integration targets
    # from the pull request or main-push change set. These jobs own CI
    # selection and receipts; they are not developer gate stages.
    "integration-plan",
    "change-owned-shard",
    "change-owned-report",
    # Stable branch-protection aggregates, not command-producing workers.
    "lint-and-unit",
    "integration",
    # Rule-id: GATE-SCOPE-WORKSPACE-AGGREGATE -- the stable aggregate merges
    # shard JUnit, checks timing, and publishes telemetry with Python. The
    # ci-fast workers own the gate.py commands.
    "integration",
    "backend-sanitizers",
    "no-ai-authorship",
    "docs",
    # Rule-id: GATE-SCOPE-CHANGES -- the changes job computes the
    # docs_only output that gates the heavy jobs' `if` (chelis#419). It
    # runs scripts/ci_detect_docs_only.py, no cargo/chelis command, so it
    # is out of gate.py scope by design.
    "changes",
    # Rule-id: GATE-SCOPE-DIAGNOSTIC-KIND -- C2.2's controlled source
    # mutations are CI-owned and run only when an owner/control path changes.
    "diagnostic-kind-oracle",
    # Rule-id: GATE-SCOPE-DTYPE-ORACLE -- the authoritative Phase 0-3
    # acceptance driver is CI-owned. It runs beside the workspace suite,
    # while the integration job below aggregates both outcomes under the
    # stable branch-protection context.
    # Rule-id: GATE-SCOPE-FAITHFUL-OBSERVATION-ORACLE -- the authoritative
    # #732 Phase 2 acceptance driver is a dedicated CI job. It runs beside
    # the workspace and dtype legs and is aggregated under the stable
    # branch-protection context.
    # Rule-id: GATE-SCOPE-COMPILED-VALUE-OWNERSHIP-ORACLE -- chelis#1286's
    # authoritative Phase 0 detector and typed expected-failure matrix is a
    # dedicated CI job aggregated under the stable integration context.
    # Rule-id: GATE-SCOPE-GENERALIZE-SWEEP-ORACLE -- chelis#1207's exact
    # sweep-versus-level parity corpus intentionally bypasses nextest's
    # default filter. Four CI-owned shards execute its disjoint partitions;
    # the aggregate retains the stable blocking status context.
    "integration",
    # Rule-id: GATE-SCOPE-TEST-TELEMETRY -- this CI-owned aggregate reads
    # nextest artifacts produced by the gate and oracle jobs. It runs no
    # cargo or Chelis command itself.
    "test-telemetry",
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
}


class RejectionAuthorityPrBoundaryTests(unittest.TestCase):
    def test_pr_ci_keeps_only_the_offline_construction_boundary(self):
        workflow = CI_YML.read_text()
        script_unit = _ci_job_block("script-unit")
        self.assertNotIn("\n  rejection-authority-liveness:", workflow)
        self.assertNotIn("rejection_authority_changed", workflow)
        self.assertNotIn("validate_rejection_issue_manifest.py", workflow)
        self.assertNotIn("issues: read", workflow)
        _assert_executable_run_once(
            script_unit,
            ".venv/bin/python scripts/check_rejection_authority_boundary.py",
        )


class DiagnosticKindOracleJobTests(unittest.TestCase):
    def test_job_is_change_gated_and_executes_the_mutation_oracle(self):
        block = _ci_job_block("diagnostic-kind-oracle")
        self.assertIn("needs: [changes]", block)
        self.assertIn("needs.changes.outputs.diagnostic_kind_changed", block)
        self.assertIn("contents: read", block)
        _assert_executable_run_once(
            block,
            ".venv/bin/python scripts/diagnostic_kind_oracle.py",
        )
        self.assertIn("taiki-e/install-action@nextest", block)

    def test_a_quoted_passing_noop_is_not_the_oracle_step(self):
        block = _ci_job_block("diagnostic-kind-oracle")
        mutated = block.replace(
            "run: .venv/bin/python scripts/diagnostic_kind_oracle.py",
            'run: "true # scripts/diagnostic_kind_oracle.py"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(
                mutated,
                ".venv/bin/python scripts/diagnostic_kind_oracle.py",
            )

# Whole WORKFLOW FILES that are out-of-scope-by-design for the per-PR developer
# `gate.py` quartet (like the backend-sanitizers / macos-workspace-shard jobs
# in ci.yml, but in their own files). They run their own commands the gate does not
# produce, by design. Listed here so the exclusion is explicit and reviewable.
# Rule-id: GATE-SCOPE-CONFORMANCE -- the Hull conformance gate runs a Python
# corpus runner against the built binary; it is a CI job, NOT part of the cargo
# quartet + lint developer gate (see tests/conformance/hull/run_conformance.py
# and .github/workflows/conformance.yml).
NON_GATE_WORKFLOWS = {
    # Mac workspace, Clippy and SMT coverage is daily/manual, outside PR CI.
    "macos-nightly.yml",
    "ci.yml",
    # PR metadata acknowledgement enforcement is intentionally isolated from
    # compiler CI, and final package expansion is an explicit exact-head
    # dispatch. Neither is a developer gate.py stage.
    "pr-contract-acknowledgements.yml",
    "pr-base-retarget.yml",
    "pr-package-expansion.yml",
    # Changelog policy uses Python only, including on docs PRs.
    "changelog.yml",
    "smt-full-prove.yml",
    "heavy-e2e.yml",
    # The manual shared-runner cache probe requires the admitted EC2 host.
    # Its commands do not belong to the per-PR developer gate.
    "shared-runner-canary.yml",
    # First separable loud-unsupported C7.5 slice: daily/manual standing
    # rejection-issue liveness on main plus tracking-issue reporting. It runs
    # no command owned by the per-PR developer gate. The broader Phase 4
    # nightly matrix remains pending.
    "loud-unsupported-nightly.yml",
    # Manually dispatched Phase 3 acceptance on a provisioned AMD GPU runner;
    # the hardware oracle is not part of the per-PR developer gate.
    "ownership-hip.yml",
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
    # OpenSpec autoland classifies an OpenSpec document change and asks
    # GitHub to merge it. It runs the stdlib-only boundary classifier and
    # the pinned central OpenSpec action, no cargo or Chelis command the
    # developer gate owns. It is an actor rather than a gate and must not
    # become a required check, so it stays outside gate.py by design.
    "openspec-autoland.yml",
    # Strict OpenSpec validation for the autoland path. Runs the pinned
    # central OpenSpec action, no cargo or Chelis command the developer
    # gate owns. Out of gate.py scope by design.
    "openspec-autoland-validate.yml",
    # The push signal and the trusted controller that reacts to it. Neither
    # runs a cargo or Chelis command the developer gate owns.
    "openspec-autoland-signal.yml",
    "openspec-autoland-controller.yml",
}


class StageUnionTests(unittest.TestCase):
    def test_full_gate_keeps_censuses_while_ci_uses_the_split_profile(self):
        self.assertIn(gate.NEXTEST_WORKSPACE_CI, gate.STAGES["integration"])
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
            set(gate.STAGES) - {"ci-fast"},
            "STAGE_ORDER lists full/manual stages, excluding the hosted fast subset",
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

    def test_lint_stage_does_not_repeat_the_workspace_build(self):
        self.assertNotIn(gate.BUILD_WORKSPACE, gate.STAGES["lint-and-unit"])
        self.assertIn(gate.CLIPPY_WORKSPACE, gate.STAGES["lint-and-unit"])

    def test_integration_partition_selects_only_the_nextest_command(self):
        commands = gate.selected_stage_commands(
            "integration",
            tests_only=True,
            support_only=False,
            partition="hash:1/2",
        )
        self.assertEqual(
            commands,
            [gate.NEXTEST_WORKSPACE_CI + ["--partition", "hash:1/2"]],
        )

    def test_integration_support_selects_each_non_test_oracle_once(self):
        commands = gate.selected_stage_commands(
            "integration",
            tests_only=False,
            support_only=True,
            partition=None,
        )
        self.assertEqual(commands, gate.STAGES["integration"][1:])

    def test_lowering_trace_runs_with_its_feature_locally_and_in_ci_support(self):
        command = ["cargo", "nextest", "run", "-p", "chelis-ir", "--features",
                   "lowering-trace", "--lib", "--test", "lowering_trace",
                   "--test", "helper_lowering_trace"]
        self.assertEqual(gate.LOWERING_TRACE_TESTS, command)
        self.assertEqual(gate.LOCAL_STATIC_COMMANDS.count(command), 1)
        commands = gate.selected_stage_commands(
            "integration", tests_only=False, support_only=True, partition=None,
        )
        self.assertEqual(commands.count(command), 1)

    def test_support_slices_preserve_every_canonical_command_once(self):
        slices = [gate.selected_stage_commands(
            "integration", tests_only=False, support_only=True,
            partition=None, support_slice=name,
        ) for name in ("frontend", "domain")]
        self.assertEqual(slices[0], [gate.LOWERING_TRACE_TESTS,
                                    gate.EMISSION_OBSERVER_TESTS,
                                    gate.COMPILER_FRONT_END_PERFORMANCE_ORACLE])
        self.assertEqual(slices[1], [
            gate.UNREPRESENTABLE_DOMAIN_ORACLE,
            [gate.MANAGED_PYTHON, "scripts/dtype_builtin_atom_closure_oracle.py"],
        ])
        self.assertEqual(slices[0] + slices[1], gate.STAGES["integration"][1:])

    def test_emission_observer_runs_with_its_feature_locally_and_in_ci_support(self):
        command = ["cargo", "nextest", "run", "-p", "chelis-compiler-api", "--features",
                   "compilation-trace,native-random-observer", "--lib", "--test", "emission_observer",
                   "--test", "execution_artifact_metadata", "--test", "compilation_trace",
                   "--test", "native_random_observer", "--test", "fixed_control_c"]
        self.assertEqual(gate.EMISSION_OBSERVER_TESTS, command)
        self.assertEqual(gate.LOCAL_STATIC_COMMANDS.count(command), 1)
        commands = gate.selected_stage_commands(
            "integration", tests_only=False, support_only=True, partition=None,
        )
        self.assertEqual(commands.count(command), 1)

    def test_support_slice_is_rejected_outside_support_only_integration(self):
        for argv in (
            ["integration", "--support-slice", "frontend"],
            ["integration", "--tests-only", "--support-slice", "frontend"],
            ["lint-and-unit", "--support-only", "--support-slice", "frontend"],
            ["integration", "--support-only", "--support-slice", "missing"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                gate.parse_args(argv)

    def test_partition_is_rejected_outside_tests_only_integration(self):
        for argv in (
            ["lint-and-unit", "--partition", "hash:1/2"],
            ["integration", "--partition", "hash:1/2"],
            ["integration", "--support-only", "--partition", "hash:1/2"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                gate.parse_args(argv)


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
        self.assertIn("cargo test -p chelis-ir --doc", rendered)
        self.assertIn("cargo test -p chelis-compiler-api --doc", rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_lint_and_unit_stage(self):
        rendered = [
            gate.render(command) for command in gate.STAGES["lint-and-unit"]
        ]
        for command in (
            "cargo test -p chelis-ir --doc",
            "cargo test -p chelis-compiler-api --doc",
            "cargo test -p chelis-pipeline-core --doc",
            "<managed-python> scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_pipeline_compile_fail_contracts_are_in_the_local_subset(self):
        rendered = [gate.render(command) for command in gate.LOCAL_STATIC_COMMANDS]
        for command in (
            "cargo test -p chelis-ir --doc",
            "cargo test -p chelis-compiler-api --doc",
            "cargo test -p chelis-pipeline-core --doc",
            "<managed-python> scripts/check_checkpoint_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_phase_b_compile_fail_is_continuous_and_local(self):
        # The ban's liveness proof. Without it, deleting the `disallowed-types`
        # entries from clippy.toml leaves every continuous job green.
        command = "<managed-python> scripts/check_hash_order_phase_b_compile_fail.py"
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["lint-and-unit"]],
        )
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
        )

    def test_configuration_closure_is_continuous_and_local(self):
        command = "<managed-python> scripts/check_configuration_closure.py"
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["lint-and-unit"]],
        )
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
        )

    def test_pipeline_core_boundary_guards_are_in_the_lint_and_unit_stage(self):
        # The dependency guard, documentation guard, and pipeline-artifact
        # compile-fail fixture must run in the per-PR gate (hosted CI runs
        # `gate.py lint-and-unit`), not only in the manual oracle.
        rendered = [
            gate.render(command) for command in gate.STAGES["lint-and-unit"]
        ]
        for command in (
            "<managed-python> scripts/pipeline_core_dependency_guard.py",
            "<managed-python> scripts/pipeline_core_documentation_guard.py",
            "<managed-python> scripts/check_pipeline_core_compile_fail.py",
        ):
            self.assertIn(command, rendered)

    def test_chelis_std_generated_artifacts_are_checked_continuously_and_locally(self):
        command = (
            "<managed-python> scripts/regenerate_chelis_std_bundle.py --debug --check"
        )
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["lint-and-unit"]],
        )
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
        )
        self.assertIn(command, [gate.render(entry) for entry in gate.full_command_list()])

    def test_unrepresentable_domain_oracle_runs_in_the_per_pr_gate(self):
        # chelis#908's "Constraint on every fix in this class": the oracle
        # must run in a continuous job. It was referenced by no workflow and
        # no gate stage, so the only thing exercising it was its own unit
        # tests, which patch the command runners; the behavioral oracle never
        # reached a compiled binary. Hosted CI runs one `gate.py <stage>` per
        # job, so membership in a stage is what makes it continuous, and
        # membership in the local subset is what makes it pre-push. Removing
        # either turns the oracle dark silently, which is the exact failure
        # mode #1089 inventories -- hence an explicit lock.
        command = "<managed-python> scripts/unrepresentable_domain_oracle.py"
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["integration"]],
        )
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
        )
        self.assertIn(command, [gate.render(entry) for entry in gate.full_command_list()])

    def test_runtime_representation_phase2_oracle_is_continuous_and_local(self):
        command = (
            "<managed-python> scripts/runtime_representation_oracle.py --phase 2"
        )
        # The oracle is a stage of its own so hosted CI can give it a runner of
        # its own; it must not also ride on the integration support slice.
        self.assertEqual(
            [gate.render(entry) for entry in gate.STAGES["runtime-representation"]],
            [command],
        )
        self.assertNotIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["integration"]],
        )
        self.assertEqual(gate.STAGE_ORDER[-1], "runtime-representation")
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
        )
        self.assertIn(command, [gate.render(entry) for entry in gate.full_command_list()])
        script = REPO_ROOT / "scripts" / "runtime_representation_oracle.py"
        self.assertTrue(script.is_file())
        self.assertIn(
            "RUNTIME REPRESENTATION PHASE 0: PASS",
            script.read_text(),
        )

    def test_front_end_performance_oracle_runs_in_continuous_integration(self):
        command = "<managed-python> scripts/compiler_front_end_performance.py"
        self.assertIn(
            command,
            [gate.render(entry) for entry in gate.STAGES["integration"]],
        )
        self.assertIn(command, [gate.render(entry) for entry in gate.full_command_list()])
        self.assertNotIn(
            command,
            [gate.render(entry) for entry in gate.LOCAL_STATIC_COMMANDS],
            "the focused crate runs are CI-owned; --local already runs each changed crate",
        )
        script = REPO_ROOT / "scripts" / "compiler_front_end_performance.py"
        self.assertTrue(script.is_file())
        self.assertIn(
            "compiler front-end performance oracle: PASS", script.read_text()
        )

    def test_the_oracles_stage_is_a_job_that_installs_nextest(self):
        # Both of the oracle's compiled obligations run `cargo nextest`. The
        # Rust-policy worker deliberately does not install it, so placing the
        # oracle in its lint-and-unit stage produces a deterministic
        # `no such command: nextest`.
        # Assert the pairing structurally: the stage the oracle lives in must
        # be run by a job that installs cargo-nextest.
        command = "<managed-python> scripts/unrepresentable_domain_oracle.py"
        owning_stages = [
            stage
            for stage, entries in gate.STAGES.items()
            if command in [gate.render(entry) for entry in entries]
        ]
        self.assertEqual(
            owning_stages,
            ["integration"],
            "the oracle must live in exactly one stage, and it must be a "
            "nextest-installing one",
        )
        workflow = CI_YML.with_name("heavy-e2e.yml").read_text()
        blocks = _workflow_job_blocks(workflow)
        owning_jobs = [
            name
            for name, block in blocks.items()
            if "gate.py integration" in block
        ]
        self.assertTrue(owning_jobs, "some job must run `gate.py integration`")
        for name in owning_jobs:
            self.assertIn(
                "taiki-e/install-action@nextest",
                blocks[name],
                f"job `{name}` runs the oracle's stage but does not install "
                "cargo-nextest",
            )

    def test_a_chelis_build_precedes_the_oracle_in_every_list_that_runs_it(self):
        # chelis#1322. The oracle drives a built `chelis` over its .dp
        # fixtures. It gets one inside the gate because a command that
        # builds that exact bin target sits earlier in the list -- a
        # property nothing stated until this lock. Two things ride on it:
        # the oracle is warm rather than cold at the point it runs, and
        # `oracle_binary_handoff` may name the built path only because the
        # list guarantees it exists. A reorder that puts the oracle first
        # would silently undo both, so assert the ordering directly rather
        # than trusting the current arrangement to stay put.
        producers = {gate.render(list(c)) for c in gate.CHELIS_BINARY_PRODUCERS}
        oracle = gate.render(gate.UNREPRESENTABLE_DOMAIN_ORACLE)
        lists = {
            "full_command_list()": gate.full_command_list(),
            "local_command_list([])": gate.local_command_list([]),
            "local_command_list(['chelis-deep'])": gate.local_command_list(
                ["chelis-deep"]
            ),
        }
        for label, commands in lists.items():
            rendered = [gate.render(command) for command in commands]
            self.assertIn(oracle, rendered, f"{label} must run the oracle")
            oracle_index = rendered.index(oracle)
            earlier = set(rendered[:oracle_index])
            self.assertTrue(
                earlier & producers,
                f"{label} runs the oracle at index {oracle_index} with no "
                f"command that builds `chelis` before it; one of {sorted(producers)} "
                "must precede it",
            )

    def test_the_handoff_variable_has_exactly_one_spelling(self):
        # chelis#1322. Two independent string literals were the most likely
        # rot channel in this whole change: rename either side and every
        # test on both sides stays green while the handoff is dead, because
        # each test refers to its own module's symbol. The oracle then sees
        # no variable, rebuilds its own binary, and still prints
        # `ORACLE: PASS` -- verbatim the silent degradation the fail-closed
        # design exists to prevent. With one constant the rename is
        # harmless, so what this asserts is that there is only one: the
        # objects are identical, and gate.py quotes no spelling of its own.
        oracle = _load_oracle_module()
        self.assertIs(gate.ORACLE_BINARY_ENV, oracle.ORACLE_BINARY_ENV)
        source = (REPO_ROOT / "scripts" / "gate.py").read_text()
        for literal in (
            f'"{oracle.ORACLE_BINARY_ENV}"',
            f"'{oracle.ORACLE_BINARY_ENV}'",
        ):
            self.assertNotIn(
                literal,
                source,
                "gate.py must import the handoff variable's spelling from "
                "the oracle, not declare a second copy of it",
            )

    def test_the_handoff_is_reported_when_a_gate_command_fails(self):
        # A failed oracle stage is debugged from the gate's diagnostic dump
        # and its printed rerun line. Both must name the handoff: without
        # it the suggested rerun builds its own binary and so does not
        # reproduce what failed.
        self.assertIn(gate.ORACLE_BINARY_ENV, gate.DIAGNOSTIC_ENVIRONMENT)
        rerun = gate._rerun_command(
            ["true"],
            {
                "PYO3_PYTHON": "/uv/python3.11",
                "CARGO_TARGET_DIR": "/w/target",
                gate.ORACLE_BINARY_ENV: "/w/target/debug/chelis",
            },
            Path("/w"),
        )
        self.assertIn(
            f"{gate.ORACLE_BINARY_ENV}=/w/target/debug/chelis", rerun
        )

    def test_the_oracle_binary_handoff_names_the_normalized_target(self):
        # The handoff must point into the same worktree-local target dir
        # `gate_environment` normalizes to, not a raw relative `target`:
        # a sibling agent's run uses target/agents/<name>, and naming the
        # wrong one would hand the oracle a stale or absent binary.
        commands = gate.local_command_list([])
        handoff = gate.oracle_binary_handoff(commands, "/w/target/agents/x")
        self.assertEqual(handoff, str(Path("/w/target/agents/x/debug/chelis")))

    def test_a_list_without_a_chelis_build_hands_over_nothing(self):
        # chelis#1322's fail-closed half. The oracle treats the variable as
        # authoritative and errors on a path that is not there, so the gate
        # may only set it where the list itself builds the binary. The
        # `integration` stage run on its own is exactly that case: hosted
        # CI's `Workspace Tests (Linux)` job reaches the oracle through
        # `cargo nextest run --workspace`, which builds the bin only as an
        # implicit consequence of chelis-cli having integration tests.
        # That is not a guarantee this list states, so CI keeps the
        # oracle's original build-it-yourself behavior.
        self.assertIsNone(
            gate.oracle_binary_handoff(gate.STAGES["integration"], "/w/target")
        )
        self.assertNotIn(
            gate.NEXTEST_WORKSPACE_CI,
            [list(c) for c in gate.CHELIS_BINARY_PRODUCERS],
        )
        self.assertNotIn(
            gate.NEXTEST_WORKSPACE,
            [list(c) for c in gate.CHELIS_BINARY_PRODUCERS],
        )

    def test_a_list_without_the_oracle_hands_over_nothing(self):
        # `gate.py lint-and-unit` builds `chelis` but never runs the
        # oracle; it has no reason to export the variable.
        self.assertIsNone(
            gate.oracle_binary_handoff(
                gate.STAGES["lint-and-unit"], "/w/target"
            )
        )

    def test_a_producer_after_the_oracle_hands_over_nothing(self):
        # The predicate itself, not the shipped lists. `oracle_binary_handoff`
        # slices `commands[:oracle_index]`; relaxing that to a membership
        # test over the whole list would name a binary that does not exist
        # yet when the oracle runs. The ordering lock above covers the three
        # lists gate.py ships, which is a different property: it would stay
        # green while the predicate degraded, because those lists happen to
        # be correctly ordered.
        self.assertIsNone(
            gate.oracle_binary_handoff(
                [gate.UNREPRESENTABLE_DOMAIN_ORACLE, gate.CHELIS_LINT_CHECK],
                "/w/target",
            )
        )
        self.assertIsNone(
            gate.oracle_binary_handoff(
                [gate.UNREPRESENTABLE_DOMAIN_ORACLE, gate.BUILD_WORKSPACE],
                "/w/target",
            )
        )
        # Positive control, so the two assertions above cannot pass merely
        # because the handoff stopped working altogether.
        self.assertIsNotNone(
            gate.oracle_binary_handoff(
                [gate.CHELIS_LINT_CHECK, gate.UNREPRESENTABLE_DOMAIN_ORACLE],
                "/w/target",
            )
        )

    def test_the_chelis_producer_set_is_exactly_the_two_unconditional_builds(
        self,
    ):
        # An exact set, not a membership check: dropping either entry leaves
        # every shipped list still handing over (the other one covers them
        # all), so nothing else would notice. Adding a command here is a
        # claim that it unconditionally leaves <target>/debug/chelis, which
        # is the assumption the fail-closed handoff rests on, so it should
        # cost a deliberate edit to this lock.
        self.assertEqual(
            gate.CHELIS_BINARY_PRODUCERS,
            (tuple(gate.BUILD_WORKSPACE), tuple(gate.CHELIS_LINT_CHECK)),
        )

    def _recorded_child_environments(self, commands, environ=None):
        """Run `commands` through `run_commands` with cargo stubbed out,
        returning the environment each child would have been given."""
        seen: list[dict[str, str]] = []

        class _Stub:
            def __init__(self) -> None:
                self.stdout = io.StringIO("")

            def wait(self) -> int:
                return 0

        # `gate_environment` probes the interpreter through subprocess.run,
        # which also reaches Popen; only the gate's own child commands
        # carry an explicit `env`, so let everything else through.
        real_popen = gate.subprocess.Popen

        def fake_popen(command, *args, **kwargs):
            if "env" not in kwargs:
                return real_popen(command, *args, **kwargs)
            seen.append(dict(kwargs["env"]))
            return _Stub()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                gate.subprocess, "Popen", side_effect=fake_popen
            ):
                result = gate.run_commands(
                    commands,
                    stage_label="handoff-test",
                    repo_root=root,
                    failure_root=root / "target/gate-failures",
                    environ=environ
                    if environ is not None
                    else {"PATH": os.environ.get("PATH", "")},
                    executable=Path(sys.executable),
                    output_stream=io.StringIO(),
                    error_stream=io.StringIO(),
                )
            self.assertEqual(result, 0)
            return seen, str((root / "target").resolve())

    def test_run_commands_exports_the_handoff_to_child_commands(self):
        # End-to-end: the computed path actually reaches the child
        # environment, which is the only place the oracle can read it.
        seen, target = self._recorded_child_environments(
            [gate.CHELIS_LINT_CHECK, gate.UNREPRESENTABLE_DOMAIN_ORACLE]
        )
        expected = str(Path(target) / "debug" / "chelis")
        self.assertEqual(len(seen), 2)
        for environment in seen:
            self.assertEqual(environment[gate.ORACLE_BINARY_ENV], expected)

    def test_run_commands_exports_no_handoff_without_a_preceding_build(self):
        seen, _ = self._recorded_child_environments(
            gate.STAGES["integration"]
        )
        self.assertEqual(len(seen), len(gate.STAGES["integration"]))
        for environment in seen:
            self.assertNotIn(gate.ORACLE_BINARY_ENV, environment)

    def test_a_bogus_explicit_handoff_aborts_before_the_first_command(self):
        # The compared discipline is PYO3_PYTHON's, and PYO3_PYTHON is
        # diagnosed at command 0 of 10. Learning about a typo only when the
        # oracle reaches it costs a whole workspace clippy, fmt, the lint
        # pass, three rustdoc stages and two guards first.
        launched: list[list[str]] = []
        # Only the gate's own child commands carry an explicit `env`;
        # `gate_environment`'s interpreter probe must still run.
        real_popen = gate.subprocess.Popen

        def record(command, *args, **kwargs):
            if "env" not in kwargs:
                return real_popen(command, *args, **kwargs)
            launched.append(list(command))
            raise AssertionError("no command may launch")

        with tempfile.TemporaryDirectory() as tmp:
            error = io.StringIO()
            with mock.patch.object(
                gate.subprocess, "Popen", side_effect=record
            ):
                result = gate.run_commands(
                    gate.local_command_list([]),
                    stage_label="bogus-handoff",
                    repo_root=Path(tmp),
                    failure_root=Path(tmp) / "target/gate-failures",
                    environ={
                        "PATH": os.environ.get("PATH", ""),
                        gate.ORACLE_BINARY_ENV: "/nope/chelis",
                    },
                    executable=Path(sys.executable),
                    output_stream=io.StringIO(),
                    error_stream=error,
                )
        self.assertEqual(result, 2)
        self.assertEqual(launched, [])
        diagnostic = error.getvalue()
        self.assertIn(gate.ORACLE_BINARY_ENV, diagnostic)
        self.assertIn("/nope/chelis", diagnostic)

    def test_the_gate_and_the_oracle_agree_that_empty_is_not_unset(self):
        # If one side read an empty value as absent and the other as a
        # handoff, an ambient `export CHELIS_ORACLE_BINARY=` would disable
        # the handoff with no notice on either side. Both reject it.
        oracle = _load_oracle_module()
        for value in ("", "   "):
            with self.subTest(value=value):
                self.assertIsNotNone(
                    gate._oracle_binary_validation_error(value, Path("/w"))
                )
                with mock.patch.dict(
                    os.environ,
                    {gate.ORACLE_BINARY_ENV: value},
                    clear=False,
                ):
                    with self.assertRaises(oracle.OracleBinaryError):
                        oracle.handed_over_binary()

    def test_a_directory_is_described_as_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            error = gate._oracle_binary_validation_error(tmp, Path("/w"))
            self.assertIsNotNone(error)
            self.assertIn("is a directory", error)

    def test_the_rerun_note_warns_that_the_pinned_binary_is_stale(self):
        # The rerun line pins the handoff so it reproduces the failure.
        # Pasted again after a Rust fix, that pin retests the old binary,
        # which in one direction reports a false pass. The diagnostic has
        # to say so, and only for the oracle's own command.
        environment = {
            "PYO3_PYTHON": sys.executable,
            "CARGO_TARGET_DIR": "/w/target",
            gate.ORACLE_BINARY_ENV: "/w/target/debug/chelis",
        }

        def diagnose(command):
            stream = io.StringIO()
            gate._print_failure_diagnostics(
                command=command,
                returncode=1,
                launch_error=None,
                duration=0.5,
                stage_label="local",
                index=10,
                total=10,
                repo_root=Path("/w"),
                failure_log=Path("/w/log"),
                environment=environment,
                tail=deque(),
                line_count=0,
                error_stream=stream,
            )
            return stream.getvalue()

        oracle_output = diagnose(
            ["/uv/python", gate.UNREPRESENTABLE_DOMAIN_ORACLE[-1]]
        )
        self.assertIn("drop that assignment", oracle_output)
        self.assertIn("retesting the old binary", oracle_output)
        # Not on an unrelated failing command in the same run.
        self.assertNotIn("drop that assignment", diagnose(gate.FMT_CHECK))

    def test_an_explicit_handoff_from_the_caller_is_not_replaced(self):
        # Same discipline `gate_environment` applies to PYO3_PYTHON: an
        # explicit setting is authoritative, so the gate does not overwrite
        # it with the binary its own commands build. It must be a usable
        # path, because the gate now validates it up front.
        chosen = sys.executable
        seen, target = self._recorded_child_environments(
            [gate.CHELIS_LINT_CHECK, gate.UNREPRESENTABLE_DOMAIN_ORACLE],
            environ={
                "PATH": os.environ.get("PATH", ""),
                gate.ORACLE_BINARY_ENV: chosen,
            },
        )
        computed = str(Path(target) / "debug" / "chelis")
        for environment in seen:
            self.assertEqual(environment[gate.ORACLE_BINARY_ENV], chosen)
            self.assertNotEqual(
                environment[gate.ORACLE_BINARY_ENV], computed
            )

    def test_unrepresentable_domain_oracle_script_exists_and_documents_acceptance(self):
        # The wiring above is worthless if the script it names is gone or
        # stops declaring its acceptance condition. #908 and the repository's
        # one-oracle rule both key on the `ORACLE: PASS` line.
        script = REPO_ROOT / "scripts" / "unrepresentable_domain_oracle.py"
        self.assertTrue(script.is_file(), f"{script} must exist")
        text = script.read_text()
        self.assertIn(
            "ORACLE: PASS",
            text,
            "the oracle must emit its documented acceptance line",
        )
        self.assertIn(
            "phase3_stamped_ingress",
            text,
            "the oracle's obligations must cover the compiler-API ingress "
            "(chelis#1088)",
        )
        self.assertIn(
            "[03-PROG-1]",
            text,
            "the oracle must check the spec/03 top-level form rule",
        )

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


_WORKFLOW_JOB_ID_PATTERN = r"[A-Za-z_][A-Za-z0-9_-]*"
_WORKFLOW_JOB_HEADER = re.compile(r"^  (?P<key>\S.*?):\s*(?:#.*)?$")
_WORKFLOW_JOBS_HEADER = re.compile(
    r"^(?:jobs|'jobs'|\"jobs\")\s*:\s*(?P<value>.*)$"
)
_WORKFLOW_STEPS_HEADER = re.compile(
    r"^(?P<indent> +)(?:steps|'steps'|\"steps\")\s*:\s*(?P<value>.*)$"
)
_UNSUPPORTED_RUN_SCALAR = "<unsupported-run-scalar>"


def _parse_workflow_jobs_header(line: str) -> bool:
    """Recognize the block-form `jobs` map and reject inline variants."""
    match = _WORKFLOW_JOBS_HEADER.fullmatch(line)
    if match is None:
        return False
    value = match.group("value").strip()
    if not value or value.startswith("#"):
        return True
    raise AssertionError(f"unsupported workflow jobs mapping: {line!r}")


def _parse_workflow_job_header(line: str) -> str | None:
    """Decode one top-level job key without silently approximating YAML.

    GitHub accepts simple quoted mapping keys as job IDs. Double-quoted YAML
    escapes need a real YAML decoder, so this dependency-free parity parser
    fails closed for that shape instead of potentially omitting a CI job.
    """
    indent = len(line) - len(line.lstrip(" "))
    content = line.strip()
    # A comment is legal YAML anywhere inside `jobs:` and is never a job id.
    # Reject it before matching, not only on the regex-miss path: the pattern's
    # `\S.*?` starts happily at `#`, so a comment ending in a colon matches and
    # would otherwise be reported as a malformed job id (chelis#1443).
    if content.startswith("#"):
        return None

    match = _WORKFLOW_JOB_HEADER.fullmatch(line)
    if match is None:
        if indent == 2 and content:
            raise AssertionError(f"unsupported workflow job entry: {line!r}")
        return None

    raw = match.group("key").strip()
    if re.fullmatch(_WORKFLOW_JOB_ID_PATTERN, raw):
        return raw

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        inner = raw[1:-1]
        if raw[0] == '"' and "\\" in inner:
            raise AssertionError(f"unsupported quoted workflow job id: {raw!r}")
        if raw[0] == "'":
            inner = inner.replace("''", "'")
        if re.fullmatch(_WORKFLOW_JOB_ID_PATTERN, inner):
            return inner
        raise AssertionError(f"unsupported quoted workflow job id: {raw!r}")

    raise AssertionError(f"unsupported workflow job id header: {raw!r}")


def _assert_supported_workflow_step_shapes(workflow: str) -> None:
    """Reject step YAML shapes the dependency-free parser cannot attribute."""
    steps_indent: int | None = None
    unsupported_entry = re.compile(r"^-\s*(?:[?{*&'\"!]|<<:)")
    unsupported_key = re.compile(r"^(?:[?:{*&'\"!]|<<:)")
    explicit_steps_key = re.compile(
        r"^\s*(?:\?\s*(?:steps|'steps'|\"steps\")|"
        r"&\S+\s+(?:steps|'steps'|\"steps\")\s*:|"
        r"\*\S+\s*:).*$"
    )

    for line in workflow.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if steps_indent is not None:
            if stripped and not stripped.startswith("#") and indent <= steps_indent:
                steps_indent = None
            else:
                relative_indent = indent - steps_indent
                if relative_indent == 2 and unsupported_entry.match(stripped):
                    raise AssertionError(
                        f"unsupported workflow step shape: {line!r}"
                    )
                if relative_indent == 4 and unsupported_key.match(stripped):
                    raise AssertionError(
                        f"unsupported workflow step shape: {line!r}"
                    )

        if steps_indent is None:
            if explicit_steps_key.fullmatch(line):
                raise AssertionError(
                    f"unsupported workflow step shape: {line!r}"
                )
            match = _WORKFLOW_STEPS_HEADER.fullmatch(line)
            if match is None:
                continue
            value = match.group("value").strip()
            if value and not value.startswith("#"):
                raise AssertionError(
                    f"unsupported workflow step shape: {line!r}"
                )
            steps_indent = len(match.group("indent"))


def _strip_yaml_scalar_quotes(command: str) -> str:
    """Remove one matching YAML quote pair and its trailing comment.

    This is deliberately a small scalar normalizer rather than a second YAML
    implementation. The parity guard only needs the shell command text, but it
    must recognize quoted scalars whose closing quote is followed by a YAML
    comment or appears on a continuation line.
    """
    command = command.strip()
    if not command or command[0] not in ("'", '"'):
        return re.sub(r"\s+#.*$", "", command).strip()

    # YAML double-quoted scalars interpret escapes and line continuations.
    # This dependency-free guard does not reproduce that grammar; reject the
    # unsupported shape so a decoded cargo/chelis command cannot disappear.
    if command[0] == '"' and "\\" in command:
        return _UNSUPPORTED_RUN_SCALAR

    quote = command[0]
    index = 1
    while index < len(command):
        if quote == "'" and command[index] == "'":
            if index + 1 < len(command) and command[index + 1] == "'":
                index += 2
                continue
            trailing = command[index + 1 :].strip()
            if not trailing or trailing.startswith("#"):
                return command[1:index].replace("''", "'").strip()
            return command
        if quote == '"':
            if command[index] == "\\":
                index += 2
                continue
            if command[index] == '"':
                trailing = command[index + 1 :].strip()
                if not trailing or trailing.startswith("#"):
                    return command[1:index].strip()
                return command
        index += 1
    return command


def _is_yaml_block_scalar(value: str) -> bool:
    if not value:
        return False
    return re.fullmatch(
        r"[|>](?:[+-][1-9]?|[1-9][+-]?)?",
        value.split(maxsplit=1)[0],
    ) is not None


def _parse_ci_run_commands(text: str | None = None) -> dict[str, list[str]]:
    """Parse `.github/workflows/ci.yml` and return every `run:` command.

    The parser is intentionally narrow line-based YAML-shape matching: it
    tracks the current `<job>:` header (two-space indent under `jobs:`), then
    reconstructs plain and quoted `run:` scalars across indented continuation
    lines. Matching outer YAML quotes and YAML comments are removed. Literal
    or folded block-scalar indicators are recorded as a sentinel. The gate
    parity lock compares this complete list with an exact per-job allowlist;
    it never tries to infer which executable Bash will ultimately run.
    """
    if text is None:
        text = CI_YML.read_text()
    _assert_supported_workflow_step_shapes(text)
    lines = text.splitlines()
    current_job: str | None = None
    invocations: dict[str, list[str]] = {}
    run_inline = re.compile(
        r"^(?P<indent>\s*)(?P<dash>-\s*)?run\s*:\s*(?P<value>.*?)\s*$"
    )
    in_jobs = False
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if _parse_workflow_jobs_header(line):
            in_jobs = True
            line_index += 1
            continue
        if not in_jobs:
            line_index += 1
            continue
        job_name = _parse_workflow_job_header(line)
        if job_name is not None:
            current_job = job_name
            invocations.setdefault(current_job, [])
            line_index += 1
            continue
        if current_job is None:
            line_index += 1
            continue
        rm = run_inline.match(line)
        if rm is None:
            line_index += 1
            continue

        scalar_parts = [rm.group("value").strip()]
        continuation_floor = len(rm.group("indent")) + len(rm.group("dash") or "")
        next_index = line_index + 1
        while next_index < len(lines):
            continuation = lines[next_index]
            if not continuation.strip():
                next_index += 1
                continue
            continuation_indent = len(continuation) - len(
                continuation.lstrip(" ")
            )
            if continuation_indent <= continuation_floor:
                break
            scalar_parts.append(continuation.strip())
            next_index += 1

        command = " ".join(part for part in scalar_parts if part).strip()
        if command.startswith(("&", "*", "!")):
            raise AssertionError(
                f"unsupported workflow run scalar: {command!r}"
            )
        if _is_yaml_block_scalar(command):
            # Literal or folded block; record a sentinel so the dedicated
            # guard test can detect every YAML-equivalent spelling.
            invocations[current_job].append("<multiline-run-block>")
            line_index = next_index
            continue
        command = _strip_yaml_scalar_quotes(command)
        invocations[current_job].append(command)
        line_index = next_index
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
    """Read a job from its explicitly registered workflow owner."""
    if job == "acknowledgements":
        return _workflow_job_block(PR_CONTRACT_ACK_YML, job)
    if job in {"package-expansion-shard", "package-expansion-summary"}:
        return _workflow_job_block(PR_PACKAGE_EXPANSION_YML, job)
    if job in {"macos-workspace-shard", "macos-smoke", "smt-build-darwin-arm64"}:
        return _workflow_job_block(CI_YML.with_name("macos-nightly.yml"), job)
    if job in {'backend-sanitizers-full', 'dtype-phase3-oracle', 'compiled-value-ownership-phase0-oracle', 'faithful-observation-phase2-oracle', 'generalize-sweep-oracle-shard', 'full-workspace', 'runtime-representation-phase0-oracle', 'integration-support', 'generalize-sweep-oracle'}:
        return _workflow_job_block(CI_YML.with_name("heavy-e2e.yml"), job)
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
            executable.append(value)
    count = executable.count(command)
    if count != 1:
        raise AssertionError(
            f"expected exactly one executable `run: {command}`, found {count}"
        )


def _ci_step_block(job_block: str, step_name: str) -> str:
    """Return one step's raw text from a job block, without the next step.

    Steps are `      - name: ...` items, so a step ends at the next line with
    that exact indentation and dash. Splitting on the marker keeps every
    continuation line, including a `run:` scalar and any `if:` condition.
    """
    marker = f"      - name: {step_name}\n"
    if job_block.count(marker) != 1:
        raise AssertionError(
            f"expected exactly one step named {step_name!r}, "
            f"found {job_block.count(marker)}"
        )
    tail = job_block[job_block.index(marker) + len(marker) :]
    end = re.search(r"^      - ", tail, re.MULTILINE)
    return tail[: end.start()] if end else tail


def _assert_support_slice_contract(block: str) -> None:
    assert "slice: [frontend, domain]" in block
    command = "python3 scripts/gate.py integration --support-only --support-slice ${{ matrix.slice }}"
    _assert_executable_run_once(block, command)
    assert block.count("--support-only") == 1
    assert "    if:" not in block


def _workflow_job_block(path: Path, job: str) -> str:
    """Return the raw workflow text block for one job."""
    block = _workflow_job_blocks(path.read_text()).get(job)
    if block is None:
        raise AssertionError(f"missing workflow job {job!r} in {path}")
    return block.rstrip("\r\n")


def _rust_cache_steps(job_block: str) -> list[dict[str, str]]:
    """Return every Swatinem/rust-cache step's `with:` inputs in a job."""
    _assert_supported_workflow_step_shapes(job_block)
    lines = job_block.splitlines()
    uses_indices = []
    uses_line = re.compile(r"(?:-\s*)?uses\s*:\s*(?P<value>.+?)\s*$")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.fullmatch(r"(?:-\s*)?uses\s*:\s*(?:#.*)?", stripped):
            raise AssertionError("unsupported workflow action reference")
        match = uses_line.fullmatch(stripped)
        if match is None:
            continue
        raw_action = match.group("value").strip()
        if raw_action.startswith(("&", "*", "!")) or _is_yaml_block_scalar(
            raw_action
        ):
            raise AssertionError(
                f"unsupported workflow action reference: {raw_action!r}"
            )
        action = _strip_yaml_scalar_quotes(raw_action)
        if raw_action[0] in {"'", '"'} and action == raw_action:
            raise AssertionError(
                f"unsupported workflow action reference: {raw_action!r}"
            )
        if action == _UNSUPPORTED_RUN_SCALAR:
            raise AssertionError(
                "unsupported quoted workflow action reference"
            )
        action_name, separator, _action_ref = action.partition("@")
        if separator and action_name.casefold() == "swatinem/rust-cache":
            uses_indices.append(idx)
    steps: list[dict[str, str]] = []
    for uses_idx in uses_indices:
        with_idx: int | None = None
        for idx in range(uses_idx + 1, len(lines)):
            if re.match(r"^\s*- ", lines[idx]):
                break
            stripped = lines[idx].strip()
            if re.match(r"^with\s*:", stripped):
                if stripped != "with:":
                    raise AssertionError(
                        "unsupported rust-cache input shape: "
                        f"{stripped!r}"
                    )
                with_idx = idx
                break
        if with_idx is None:
            steps.append({})
            continue

        inputs: dict[str, str] = {}
        for line in lines[with_idx + 1 :]:
            if re.match(r"^\s*- ", line):
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.fullmatch(
                r"(?P<key>[A-Za-z0-9_-]+)\s*:\s*(?P<value>.+?)",
                stripped,
            )
            if match is None:
                raise AssertionError(
                    "unsupported rust-cache input shape: "
                    f"{stripped!r}"
                )
            key = match.group("key")
            raw_value = match.group("value").strip()
            if (
                raw_value.startswith(("&", "*", "!", "{", "[", "#"))
                or _is_yaml_block_scalar(raw_value)
            ):
                raise AssertionError(
                    "unsupported rust-cache input shape: "
                    f"{stripped!r}"
                )
            value = _strip_yaml_scalar_quotes(raw_value)
            if value == _UNSUPPORTED_RUN_SCALAR or (
                raw_value.startswith(("'", '"')) and value == raw_value
            ):
                raise AssertionError(
                    "unsupported rust-cache input shape: "
                    f"{stripped!r}"
                )
            if not value or key in inputs:
                raise AssertionError(
                    "unsupported rust-cache input shape: "
                    f"{stripped!r}"
                )
            if key == "shared-key" and ("${{" in value or "}}" in value):
                raise AssertionError(
                    "rust-cache shared-key must be a literal scalar: "
                    f"{stripped!r}"
                )
            inputs[key] = value
        steps.append(inputs)
    return steps


def _rust_cache_inputs(job_block: str) -> dict[str, str]:
    """Return inputs when a job has exactly one rust-cache step."""
    steps = _rust_cache_steps(job_block)
    if len(steps) != 1:
        raise AssertionError(
            "expected exactly one Swatinem/rust-cache@v2 step, "
            f"found {len(steps)}"
        )
    return steps[0]


def _assert_shared_rust_cache_writer_contract(workflow: str, *, macos: bool = False) -> None:
    """Require exactly one reviewed writer for each shared cache namespace."""
    census: list[tuple[str, dict[str, str]]] = []
    for job, block in _workflow_job_blocks(workflow).items():
        census.extend((job, inputs) for inputs in _rust_cache_steps(block))

    expected = {
        "linux-workspace": (
            "ci-fast",
            "${{ github.ref == 'refs/heads/main' }}",
        ),
        "macos-workspace": (
            "macos-workspace-shard",
            "${{ matrix.shard == 1 }}",
        ),
    }
    expected = {key: value for key, value in expected.items() if (key == "macos-workspace") == macos}
    for shared_key, expected_writer in expected.items():
        namespace = [
            (job, inputs)
            for job, inputs in census
            if inputs.get("shared-key") == shared_key
        ]
        writers = [
            (job, inputs.get("save-if", "true"))
            for job, inputs in namespace
            if inputs.get("save-if", "true") != "false"
        ]
        if len(writers) != 1:
            raise AssertionError(
                f"{shared_key} must have exactly one writer across the "
                f"workflow, found {writers}"
            )
        if writers[0] != expected_writer:
            raise AssertionError(
                f"{shared_key} writer must be {expected_writer}, "
                f"found {writers[0]}"
            )


def _assert_hash_partition_contract(
    shard_block: str, *, expected_count: int
) -> None:
    matrix_match = re.search(r"^\s+shard:\s*\[([^]]+)\]$", shard_block, re.M)
    if matrix_match is None:
        raise AssertionError("missing explicit shard matrix")
    shards = [int(value.strip()) for value in matrix_match.group(1).split(",")]
    partition_match = re.search(
        r"--partition hash:\$\{\{ matrix\.shard \}\}/(\d+)",
        shard_block,
    )
    if partition_match is None:
        raise AssertionError("missing hash partition command")
    partition_count = int(partition_match.group(1))
    if partition_count != expected_count:
        raise AssertionError(
            f"expected {expected_count} partitions, found {partition_count}"
        )
    expected = list(range(1, partition_count + 1))
    if shards != expected:
        raise AssertionError(
            "the shard matrix must cover every nextest hash partition exactly "
            f"once: expected {expected}, found {shards}"
        )


def _assert_generalize_sweep_partition_contract(shard_block: str) -> None:
    _assert_hash_partition_contract(shard_block, expected_count=4)


def _assert_read_only_workspace_cache(job_block: str) -> None:
    inputs = _rust_cache_inputs(job_block)
    if inputs.get("shared-key") != "linux-workspace":
        raise AssertionError("read-only job does not use the workspace cache")
    if inputs.get("save-if") != "false":
        raise AssertionError("read-only job must not save a competing cache entry")
    if "CARGO_TARGET_DIR:" in job_block:
        raise AssertionError(
            "an explicit target path changes rust-cache's environment hash and "
            "silently prevents reuse of the workspace cache"
        )


class CiParityTests(unittest.TestCase):
    """Lock the complete CI topology around the canonical developer gate."""

    def test_ci_file_exists(self):
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")

    def test_gate_jobs_call_gate_py(self):
        # Every run scalar in a gate-owned worker is reviewed here. This is a
        # structural contract, not a partial Bash executable classifier.
        commands = _parse_ci_run_commands()
        for job, expected in GATE_WORKER_RUN_COMMANDS.items():
            self.assertIn(job, commands, f"CI job '{job}' not found")
            self.assertEqual(
                commands[job],
                list(expected),
                (
                    f"CI job '{job}' has unreviewed or reordered run commands; "
                    "route gate work through scripts/gate.py and review any "
                    "bootstrap or reporting command explicitly"
                ),
            )
        # Positive parity: the workflow text must actually call
        # gate.py for every stage. The workspace test shards select only the
        # partitioned nextest command; shard 2 selects the remaining
        # integration oracles exactly once after its test partition; the
        # runtime-representation stage has a worker of its own.
        text = CI_YML.read_text()
        self.assertIn("scripts/gate.py lint-and-unit", text)
        self.assertIn(
            "scripts/gate.py ci-fast",
            text,
        )
        self.assertIn("scripts/gate.py integration --support-only", CI_YML.with_name("heavy-e2e.yml").read_text())
        self.assertIn("scripts/gate.py runtime-representation", CI_YML.with_name("heavy-e2e.yml").read_text())

    def test_python_binding_ingress_suite_is_continuous(self):
        text = CI_YML.read_text()
        self.assertIn(
            ".venv/bin/python -m unittest discover -s bindings/python/tests "
            "-p 'test_*.py'",
            text,
            (
                "bindings/python/tests contains the #729 Python-ingress oracle; "
                "the script-unit job must discover it continuously"
            ),
        )
        block = _ci_job_block("script-unit")
        dependencies = "uv pip install --python .venv/bin/python -r bindings/python/pyproject.toml"
        scripts = ".venv/bin/python scripts/ci_script_tests.py pr"
        bindings = ".venv/bin/python -m unittest discover -s bindings/python/tests -p 'test_*.py'"
        for command in (dependencies, scripts, bindings):
            _assert_executable_run_once(block, command)
        for command in (scripts, bindings):
            self.assertLess(
                block.index("run: " + dependencies),
                block.index("run: " + command),
                "binding dependencies must precede every native facade consumer",
            )

    def test_native_binding_dependencies_cannot_follow_script_discovery(self):
        block = _ci_job_block("script-unit")
        dependency_step = (
            "      - name: Install Python binding dependencies\n"
            "        run: uv pip install --python .venv/bin/python -r bindings/python/pyproject.toml\n\n"
        )
        self.assertEqual(block.count(dependency_step), 1)
        late = block.replace(dependency_step, "").replace(
            "      - name: Python binding ingress unit tests",
            dependency_step + "      - name: Python binding ingress unit tests",
        )
        with mock.patch(__name__ + "._ci_job_block", return_value=late):
            with self.assertRaisesRegex(AssertionError, "binding dependencies"):
                self.test_python_binding_ingress_suite_is_continuous()

    def test_dtype_phase3_oracle_runs_beside_the_workspace_suite(self):
        workspace_block = _ci_job_block("integration")
        oracle_block = _ci_job_block("dtype-phase3-oracle")
        aggregate_block = _ci_job_block("integration")
        workflow = CI_YML.read_text()
        top_level_permissions = workflow[
            workflow.index("permissions:\n") : workflow.index("\nenv:\n")
        ]
        numpy_command = "run: uv pip install --python .venv/bin/python -r bindings/python/pyproject.toml"
        oracle_command = "run: .venv/bin/python scripts/dtype_phase3_oracle.py"
        authenticated_oracle = (
            "env:\n"
            "          GH_TOKEN: ${{ github.token }}\n"
            "          CHELIS_TEST_SHARED_REEF_HOME: "
            "${{ runner.temp }}/chelis-test-shared-reef\n"
            f"        {oracle_command}"
        )
        self.assertNotIn("  issues: read", top_level_permissions)
        self.assertNotIn(oracle_command, workspace_block)
        self.assertEqual(oracle_block.count("    contents: read"), 1)
        self.assertEqual(oracle_block.count("    issues: read"), 1)
        self.assertEqual(oracle_block.count(numpy_command), 1)
        _assert_executable_run_once(oracle_block, oracle_command.removeprefix("run: "))
        self.assertEqual(oracle_block.count(authenticated_oracle), 1)
        self.assertLess(
            oracle_block.index(numpy_command),
            oracle_block.index(oracle_command),
        )
        self.assertIn(
            "needs: [changes, ci-fast, change-owned-report]",
            workspace_block,
        )
        self.assertIn(
            "if: github.event_name == 'push'",
            workspace_block,
        )
        self.assertIn(
            "if: github.event_name != 'push'",
            workspace_block,
        )
        self.assertNotIn("    needs:", oracle_block)
        self.assertEqual(oracle_block.count("    needs:"), 0)
        self.assertNotIn("needs.integration", oracle_block)
        self.assertNotIn("dtype-phase3-oracle", workspace_block)
        self.assertIn("name: Integration Tests (Linux)", aggregate_block)
        self.assertIn(
            "needs: [changes, ci-fast, change-owned-report]",
            aggregate_block,
        )
        self.assertIn("always()", aggregate_block)
        self.assertNotIn("!cancelled()", aggregate_block)
        self.assertIn("scripts/ci_require_success.py", aggregate_block)

    def test_faithful_observation_phase2_oracle_is_a_dedicated_blocking_job(self):
        workspace_block = _ci_job_block("integration")
        dtype_block = _ci_job_block("dtype-phase3-oracle")
        oracle_block = _ci_job_block("faithful-observation-phase2-oracle")
        aggregate_block = _ci_job_block("integration")
        command = ".venv/bin/python scripts/faithful_observation_phase2_oracle.py"

        self.assertIn("name: Faithful Observation Phase 2 Oracle", oracle_block)
        self.assertNotIn("    needs:", oracle_block)
        self.assertIn("contents: read", oracle_block)
        self.assertIn("dtolnay/rust-toolchain@stable", oracle_block)
        self.assertIn("python3 scripts/ci_setup_uv_python.py", oracle_block)
        self.assertIn("taiki-e/install-action@nextest", oracle_block)
        cache_inputs = _rust_cache_inputs(oracle_block)
        self.assertEqual(cache_inputs.get("shared-key"), "linux-workspace")
        self.assertEqual(cache_inputs.get("save-if"), "false")
        self.assertNotIn("CARGO_TARGET_DIR:", oracle_block)
        _assert_executable_run_once(oracle_block, command)
        self.assertNotIn(command, workspace_block)
        self.assertNotIn(command, dtype_block)
        self.assertNotIn("faithful-observation-phase2-oracle", aggregate_block)
        self.assertIn("faithful-observation-phase2-oracle", _workflow_job_block(CI_YML.with_name("heavy-e2e.yml"), "report"))

    def test_compiled_value_ownership_stable_job_invokes_phase2_and_launch_oracles(self):
        workspace_block = _ci_job_block("integration")
        dtype_block = _ci_job_block("dtype-phase3-oracle")
        faithful_block = _ci_job_block("faithful-observation-phase2-oracle")
        oracle_block = _ci_job_block("compiled-value-ownership-phase0-oracle")
        aggregate_block = _ci_job_block("integration")
        commands = (
            ".venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2",
            ".venv/bin/python scripts/compiled_value_ownership_oracle.py --phase launch",
        )

        self.assertIn(
            "name: Compiled Value Ownership Phase 2 Oracle",
            oracle_block,
        )
        self.assertNotIn("    needs:", oracle_block)
        self.assertIn("contents: read", oracle_block)
        self.assertIn("dtolnay/rust-toolchain@stable", oracle_block)
        self.assertIn("python3 scripts/ci_setup_uv_python.py", oracle_block)
        cache_inputs = _rust_cache_inputs(oracle_block)
        self.assertEqual(cache_inputs.get("shared-key"), "linux-workspace")
        self.assertEqual(cache_inputs.get("save-if"), "false")
        self.assertNotIn("CARGO_TARGET_DIR:", oracle_block)
        for command in commands:
            _assert_executable_run_once(oracle_block, command)
            self.assertNotIn(command, workspace_block)
            self.assertNotIn(command, dtype_block)
            self.assertNotIn(command, faithful_block)
        self.assertNotIn("compiled-value-ownership-phase0-oracle", aggregate_block)
        self.assertIn("compiled-value-ownership-phase0-oracle", _workflow_job_block(CI_YML.with_name("heavy-e2e.yml"), "report"))

    def test_runtime_representation_phase2_oracle_is_a_dedicated_gate_job(self):
        workspace_block = _ci_job_block("ci-fast")
        ownership_block = _ci_job_block("compiled-value-ownership-phase0-oracle")
        oracle_block = _ci_job_block("runtime-representation-phase0-oracle")
        aggregate_block = _ci_job_block("integration")
        command = "python3 scripts/gate.py runtime-representation"

        self.assertIn(
            "name: Runtime Representation Phase 2 Oracle",
            oracle_block,
        )
        self.assertNotIn("    needs:", oracle_block)
        self.assertIn("contents: read", oracle_block)
        self.assertIn(
            "name: runtime-representation-phase2-receipts",
            oracle_block,
        )
        self.assertIn(
            "target/runtime-representation-phase1/",
            oracle_block,
        )
        self.assertIn(
            "target/runtime-representation-phase2/",
            oracle_block,
        )
        self.assertIn("dtolnay/rust-toolchain@stable", oracle_block)
        self.assertIn("python3 scripts/ci_setup_uv_python.py", oracle_block)
        self.assertIn("taiki-e/install-action@nextest", oracle_block)
        # The oracle's C/Objective-C header lanes parse through clang.
        self.assertIn(
            "python3 scripts/ci_apt_get.py gcc clang libopenblas-dev "
            "libasan8 libubsan1",
            oracle_block,
        )
        _assert_read_only_workspace_cache(oracle_block)
        _assert_executable_run_once(oracle_block, command)
        # The stage runs in exactly one job: not on the workspace shards, whose
        # support slice it left, and not folded into another oracle's job.
        self.assertNotIn(command, workspace_block)
        self.assertNotIn(command, ownership_block)
        self.assertNotIn("runtime_representation_oracle.py", workspace_block)
        self.assertNotIn("runtime-representation-phase0-oracle", aggregate_block)
        self.assertIn("runtime-representation-phase0-oracle", _workflow_job_block(CI_YML.with_name("heavy-e2e.yml"), "report"))

    def test_generalize_sweep_oracle_is_a_sharded_blocking_aggregate(self):
        shard_block = _ci_job_block("generalize-sweep-oracle-shard")
        oracle_block = _ci_job_block("generalize-sweep-oracle")
        aggregate_block = _ci_job_block("integration")
        command = (
            "cargo nextest run --workspace --profile ci-full "
            "--ignore-default-filter "
            "--features chelis-types/generalize-sweep-oracle --no-fail-fast "
            "-E 'not (binary_id(/^chelis-compiler-api::capacity_census_wire$/) | "
            "binary_id(/^chelis-python::capacity_census_bindings$/) | "
            "binary_id(/^chelis-cli::stdlib_typecheck_cache_concurrency$/) "
            "| (binary_id(/^chelis-cli::issue_1293_redteam_round4$/) "
            "& test(/^recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c$/)))' "
            "--partition hash:${{ matrix.shard }}/4"
        )

        self.assertIn(
            "name: Typecheck Level Generalization Oracle "
            "(shard ${{ matrix.shard }}/4)",
            shard_block,
        )
        self.assertIn("fail-fast: false", shard_block)
        self.assertIn("shard: [1, 2, 3, 4]", shard_block)
        self.assertNotIn("    needs:", shard_block)
        self.assertIn("contents: read", shard_block)
        self.assertIn("dtolnay/rust-toolchain@stable", shard_block)
        self.assertIn("python3 scripts/ci_setup_uv_python.py", shard_block)
        self.assertIn("taiki-e/install-action@nextest", shard_block)
        _assert_executable_run_once(shard_block, command)

        self.assertIn("name: Typecheck Level Generalization Oracle", oracle_block)
        self.assertIn(
            "needs: [generalize-sweep-oracle-shard]",
            oracle_block,
        )
        self.assertIn("contents: read", oracle_block)
        self.assertNotIn("cargo nextest", oracle_block)
        _assert_executable_run_once(
            oracle_block,
            "python3 scripts/ci_require_success.py "
            "generalize-sweep-oracle-shard="
            "${{ needs.generalize-sweep-oracle-shard.result }}",
        )
        self.assertNotIn("generalize-sweep-oracle", aggregate_block)
        self.assertIn("generalize-sweep-oracle", _workflow_job_block(CI_YML.with_name("heavy-e2e.yml"), "report"))

    def test_generalize_sweep_partition_contract_has_no_gap_or_overlap(self):
        shard_block = _ci_job_block("generalize-sweep-oracle-shard")
        _assert_generalize_sweep_partition_contract(shard_block)

    def test_generalize_sweep_partition_contract_rejects_a_missing_shard(self):
        shard_block = _ci_job_block("generalize-sweep-oracle-shard")
        mutated = shard_block.replace(
            "shard: [1, 2, 3, 4]", "shard: [1, 2, 3]", 1
        )
        with self.assertRaisesRegex(
            AssertionError,
            "cover every nextest hash partition exactly once",
        ):
            _assert_generalize_sweep_partition_contract(mutated)

    def test_fast_worker_and_nightly_support_have_separate_owners(self):
        worker = _ci_job_block("ci-fast")
        self.assertNotIn("    strategy:", worker)
        self.assertIn("scripts/gate.py ci-fast", worker)
        self.assertNotIn("--support-only", worker)
        self.assertNotIn("ProfilePartitionTests", worker)
        _assert_support_slice_contract(_ci_job_block("integration-support"))
        self.assertIn(
            "needs: [changes, ci-fast, change-owned-report]",
            _ci_job_block("integration"),
        )

    def test_support_slice_contract_rejects_missing_repeated_or_skipped_work(self):
        block = _ci_job_block("integration-support")
        mutations = (
            block.replace("slice: [frontend, domain]", "slice: [frontend, frontend]"),
            block.replace(" --support-slice ${{ matrix.slice }}", ""),
            block.replace("    steps:", "    if: false\n    steps:"),
            block + "      - run: python3 scripts/gate.py integration --support-only --support-slice ${{ matrix.slice }}\n",
        )
        for changed in mutations:
            with self.subTest(workflow=changed), self.assertRaises(AssertionError):
                _assert_support_slice_contract(changed)

    def test_fast_worker_publishes_junit_and_selection_receipts(self):
        worker = _ci_job_block("ci-fast")
        self.assertIn("name: junit-linux-fast", worker)
        self.assertIn("path: target/nextest/ci-fast/junit.xml", worker)
        self.assertIn("name: ci-fast-receipts", worker)
        self.assertIn("target/ci-fast", worker)
        self.assertIn("target/gate-reports", worker)
        self.assertNotIn("continue-on-error: true", worker)

    def test_every_partitioned_test_lane_publishes_named_junit(self):
        expectations = {
            "ci-fast": (
                "junit-linux-fast",
                "target/nextest/ci-fast/junit.xml",
            ),
            "dtype-phase3-oracle": (
                "junit-linux-dtype",
                "target/nextest/ci-full/junit.xml",
            ),
            "generalize-sweep-oracle-shard": (
                "junit-linux-generalization-${{ matrix.shard }}",
                "target/nextest/ci-full/junit.xml",
            ),
            # chelis#1819: the nightly workspace suite is a four-way hash
            # partition, so its report is one artifact per shard. A shared
            # name would collide on upload and strand the telemetry download.
            "full-workspace": (
                "junit-linux-full-${{ matrix.shard }}",
                "target/nextest/ci-full/junit.xml",
            ),
            "macos-workspace-shard": (
                "junit-macos-workspace-${{ matrix.shard }}",
                "target/nextest/ci-full/junit.xml",
            ),
        }
        for job, (artifact, path) in expectations.items():
            with self.subTest(job=job):
                block = _ci_job_block(job)
                self.assertEqual(block.count("uses: actions/upload-artifact@v7"), 2 if job in {"ci-fast", "dtype-phase3-oracle"} else 1)
                self.assertIn(f"name: {artifact}", block)
                self.assertIn(f"path: {path}", block)
                self.assertIn("if-no-files-found: error", block)

    def test_every_nonworkspace_pr_lane_validates_and_reports_timing(self):
        expectations = {
            "dtype-phase3-oracle": "target/nextest/ci-full/junit.xml",
            "generalize-sweep-oracle-shard": (
                "target/nextest/ci-full/junit.xml"
            ),
            "macos-workspace-shard": "target/nextest/ci-full/junit.xml",
        }
        for job, junit in expectations.items():
            with self.subTest(job=job):
                block = _ci_job_block(job)
                self.assertEqual(
                    block.count("name: Validate and report test timing"), 1
                )
                self.assertIn("scripts/test_timing_check.py", block)
                self.assertIn(f"--junit {junit}", block)
                self.assertRegex(block, r"(?m)^\s+--informational\s*$")
                self.assertNotIn("--informational-relative", block)
                self.assertNotIn("continue-on-error: true", block)

    def test_cross_lane_telemetry_requires_every_expected_artifact(self):
        block = _ci_job_block("test-telemetry")
        self.assertIn("name: CI Test Telemetry", block)
        self.assertIn(
            "needs: [changes, ci-fast]",
            block,
        )
        self.assertEqual(block.count("uses: actions/download-artifact@v7"), 1)
        for artifact in (
            "junit-linux-fast",
        ):
            self.assertIn(f"name: {artifact}", block)
        self.assertIn("scripts/ci_test_telemetry.py", block)
        self.assertNotIn("--require-disjoint", block)
        self.assertIn("uses: actions/upload-artifact@v7", block)

    def test_macos_suite_is_two_disjoint_shards_behind_stable_aggregate(self):
        shard_block = _ci_job_block("macos-workspace-shard")
        aggregate_block = _ci_job_block("macos-smoke")
        _assert_hash_partition_contract(shard_block, expected_count=2)
        self.assertIn("cargo nextest run --workspace", shard_block)
        self.assertIn("--partition hash:${{ matrix.shard }}/2", shard_block)
        self.assertIn("if: matrix.shard == 2", shard_block)
        self.assertIn("name: macOS Smoke", aggregate_block)
        self.assertIn(
            "needs: [macos-workspace-shard]", aggregate_block
        )
        self.assertIn("scripts/ci_require_success.py", aggregate_block)

    def test_topology_docs_name_current_shard_owners(self):
        doc = (REPO_ROOT / "docs/ci_validation.md").read_text()
        for owner in ("ci-fast", ".config/ci-test-targets.toml", "heavy-e2e.yml", "macos-nightly.yml"):
            self.assertIn(owner, doc)
        self.assertIn("not a phase acceptance result", doc)
        self.assertIn("--ignore-default-filter", doc)


    def test_topology_docs_guard_runs_in_docs_job(self):
        docs_block = _ci_job_block("docs")
        step = _ci_step_block(docs_block, "Validate CI topology documentation")
        entrypoint = re.search(r"-m unittest (\S+)", step).group(1)
        loader = unittest.TestLoader()
        loader.loadTestsFromName(entrypoint)
        self.assertEqual(loader.errors, [], "the hosted topology guard must resolve")
        self.assertIn("uses: astral-sh/setup-uv@v8.1.0", docs_block)
        self.assertIn(
            "uv run --managed-python --python 3.11 --no-project python "
            "-m unittest scripts.test_gate.CiParityTests."
            "test_topology_docs_name_current_shard_owners",
            docs_block,
        )

    def test_phase4b_freeze_oracle_runs_in_the_always_run_docs_job(self):
        # chelis#729's Phase 4B freeze is enforced by document digests over
        # spec/ and spec/design/. A docs-only pull request skips every heavy
        # job, so the oracle has to live in an always-run job or the edits most
        # likely to break the freeze are the ones nothing checks.
        docs_block = _ci_job_block("docs")
        command = (
            "uv run --managed-python --python 3.11 --no-project python "
            "scripts/dtype_phase4b_oracle.py"
        )
        _assert_executable_run_once(docs_block, command)
        self.assertIn("uses: astral-sh/setup-uv@v8.1.0", docs_block)
        self.assertLess(
            docs_block.index(command),
            docs_block.index("run: mdbook build docs/book"),
            "the freeze oracle must fail before the slower mdBook build",
        )
        self.assertNotIn(
            "dtype_phase4b_oracle.py",
            _ci_job_block("changes"),
            "the docs-only detector must not gate the freeze oracle",
        )
        # A job-level `if:` is caught by test_always_run_jobs_are_not_gated, but
        # a STEP-level one is not, and the step is where a future editor would
        # add `docs_only` -- the exact evasion this wiring exists to prevent.
        step = _ci_step_block(docs_block, "Validate the chelis#729 Phase 4B freeze")
        conditioned = [
            line for line in step.splitlines() if re.match(r"^\s+if:", line)
        ]
        self.assertEqual(
            conditioned,
            [],
            f"the freeze step must run unconditionally; found {conditioned}",
        )

    def test_phase4b_acknowledgement_gate_is_enforced_on_pull_requests(self):
        # The whole-file digest table used to be the additive-contradiction
        # gate, and it was checked on every event. Its replacement needs a pull
        # request body, so the enforcing run is a separate step gated on the
        # event -- never on `docs_only`, which is the evasion the unconditional
        # step above exists to prevent.
        acknowledgement_block = _ci_job_block("acknowledgements")
        command = (
            '.venv/bin/python scripts/phase4b_change_report.py --pr-head "$PR_HEAD" '
            "--require-acknowledgement --acknowledgements-env PR_BODY "
            "--output target/phase4b-contract-changes.json"
        )
        _assert_executable_run_once(acknowledgement_block, command)
        step = _ci_step_block(
            acknowledgement_block, "Require frozen contract acknowledgements"
        )
        conditions = [
            line.split("if:", 1)[1].strip()
            for line in step.splitlines()
            if re.match(r"^\s+if:", line)
        ]
        self.assertEqual(conditions, ["always()"])
        # The body is attacker-controlled text. It reaches the oracle through
        # the environment, so it is never interpolated into a shell command.
        self.assertIn("PR_BODY: ${{ github.event.pull_request.body }}", step)
        # The actual synthetic merge is validated against the event head, then
        # its first parent supplies both acknowledgement comparisons.
        self.assertIn("PR_HEAD: ${{ github.event.pull_request.head.sha }}", step)
        self.assertNotIn("base.sha", step)
        self.assertNotIn("--base", step)
        # A step-level `continue-on-error: true` would leave the command pin
        # above matching while the gate stopped gating.
        self.assertNotIn("continue-on-error", step)
        self.assertNotIn("${{ github.event.pull_request.body }}", step.split("run:")[-1])

    def test_docs_checkout_is_deep_enough_for_the_merge_base(self):
        # The acknowledgement gate diffs against the merge base with the base
        # branch. A shallow checkout cannot compute one, and the gate fails
        # loudly rather than passing, so the depth is part of the wiring.
        docs_block = _ci_job_block("docs")
        checkout = docs_block[: docs_block.index("- name: Free disk space")]
        self.assertIn("uses: actions/checkout@v6", checkout)
        self.assertIn("fetch-depth: 0", checkout)

    def test_commented_phase4b_oracle_is_not_an_executable_step(self):
        block = _ci_job_block("docs")
        command = (
            "uv run --managed-python --python 3.11 --no-project python "
            "scripts/dtype_phase4b_oracle.py"
        )
        mutated = block.replace(
            f"run: {command}",
            f'# run: {command}\n        run: "true"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(mutated, command)

    def test_parallel_jobs_share_one_saved_rust_cache_namespace(self):
        _assert_shared_rust_cache_writer_contract(CI_YML.read_text())
        _assert_shared_rust_cache_writer_contract(CI_YML.with_name("macos-nightly.yml").read_text(), macos=True)
        workspace_inputs = _rust_cache_inputs(
            _ci_job_block("ci-fast")
        )
        read_only_jobs = (
            "change-owned-shard",
            "package-expansion-shard",
            "dtype-phase3-oracle",
            "faithful-observation-phase2-oracle",
            "compiled-value-ownership-phase0-oracle",
            "runtime-representation-phase0-oracle",
            "generalize-sweep-oracle-shard",
        )
        self.assertEqual(workspace_inputs.get("shared-key"), "linux-workspace")
        self.assertEqual(
            workspace_inputs.get("save-if"), "${{ github.ref == 'refs/heads/main' }}"
        )
        macos_inputs = _rust_cache_inputs(
            _ci_job_block("macos-workspace-shard")
        )
        self.assertEqual(macos_inputs.get("shared-key"), "macos-workspace")
        self.assertEqual(
            macos_inputs.get("save-if"), "${{ matrix.shard == 1 }}"
        )
        for job in read_only_jobs:
            with self.subTest(job=job):
                block = _ci_job_block(job)
                _assert_read_only_workspace_cache(block)

    def test_capacity_rustdoc_cache_is_restored_by_its_linux_owner(self):
        path = "path: target/agents/729-capacity-rustdoc"
        key = (
            "key: ${{ runner.os }}-${{ runner.arch }}-capacity-rustdoc-v1-"
            "${{ hashFiles('Cargo.lock', 'rust-toolchain.toml', "
            "'scripts/capacity_census_typed.py', "
            "'crates/chelis-compiler-api/**', 'crates/chelis-python/**') }}"
        )
        dtype = _ci_job_block("dtype-phase3-oracle")
        generalization = _ci_job_block("generalize-sweep-oracle-shard")
        for name, block in (
            ("dtype-phase3-oracle", dtype),
        ):
            with self.subTest(job=name):
                self.assertEqual(block.count("uses: actions/cache/restore@v4"), 2)
                self.assertIn(path, block)
                self.assertIn(key, block)
        self.assertEqual(dtype.count("uses: actions/cache/save@v4"), 2)
        self.assertNotIn("github.event_name == 'push'", dtype)
        self.assertIn("github.ref == 'refs/heads/main'", dtype)
        self.assertNotIn("uses: actions/cache/save@v4", generalization)
        self.assertNotIn("uses: actions/cache/restore@v4", generalization)

    def test_nextest_jobs_share_one_reef_fixture_root_per_runner(self):
        setting = (
            "CHELIS_TEST_SHARED_REEF_HOME: "
            "${{ runner.temp }}/chelis-test-shared-reef"
        )
        execution_steps = {
            "ci-fast": "Gate (units and reviewed integrations)",
            "change-owned-shard": "Execute the exact change-owned shard",
            "package-expansion-shard": (
                "Execute the informational package-expansion shard"
            ),
            "dtype-phase3-oracle": "Dtype Phase 0-3 oracle",
            "faithful-observation-phase2-oracle": (
                "Faithful observation Phase 2 oracle"
            ),
            "generalize-sweep-oracle-shard": (
                "Typecheck level generalization oracle"
            ),
            "macos-workspace-shard": "Workspace tests",
        }
        for job, step in execution_steps.items():
            with self.subTest(job=job):
                block = _ci_job_block(job)
                self.assertEqual(block.count(setting), 1)
                step_start = block.index(f"- name: {step}\n")
                step_end = block.find("\n      - name:", step_start + 1)
                if step_end == -1:
                    step_end = len(block)
                execution_step = block[step_start:step_end]
                self.assertIn("\n        env:\n", execution_step)
                self.assertIn(setting, execution_step)
    def test_shared_cache_writer_contract_censuses_every_ci_job(self):
        workflow = CI_YML.read_text()
        competing_step = (
            "\n      - name: Competing cache writer\n"
            "        uses: Swatinem/rust-cache@v2\n"
            "        with:\n"
            "          shared-key: linux-workspace\n"
            "          save-if: true\n"
        )
        mutated = workflow.replace(
            "\n  ci-fast:",
            competing_step + "\n  ci-fast:",
            1,
        )
        self.assertNotEqual(mutated, workflow, "mutation did not apply")
        with self.assertRaisesRegex(
            AssertionError,
            "linux-workspace.*exactly one writer",
        ):
            _assert_shared_rust_cache_writer_contract(mutated)

    def test_cache_input_parser_normalizes_or_rejects_yaml_equivalents(self):
        cases = (
            (
                '        with:\n          shared-key: "linux-workspace"\n'
                "          save-if: true\n",
                "exactly one writer",
            ),
            (
                "        with:\n          shared-key: 'linux-workspace'\n"
                "          save-if: true\n",
                "exactly one writer",
            ),
            (
                "        with:\n          shared-key: linux-workspace # writer\n"
                "          save-if: true\n",
                "exactly one writer",
            ),
            (
                "        with:\n          shared-key : linux-workspace\n"
                "          save-if: true\n",
                "exactly one writer",
            ),
            (
                "        with:\n          shared-key: &workspace_key "
                "linux-workspace\n          save-if: true\n",
                "unsupported rust-cache input shape",
            ),
            (
                "        with:\n          shared-key: !!str linux-workspace\n"
                "          save-if: true\n",
                "unsupported rust-cache input shape",
            ),
            (
                "        with:\n          shared-key: >-\n"
                "            linux-workspace\n          save-if: true\n",
                "unsupported rust-cache input shape",
            ),
            (
                '        with:\n          "shared-key": linux-workspace\n'
                "          save-if: true\n",
                "unsupported rust-cache input shape",
            ),
            (
                "        with:\n          ? shared-key\n"
                "          : linux-workspace\n          save-if: true\n",
                "unsupported rust-cache input shape",
            ),
            (
                "        with: {shared-key: linux-workspace, "
                "save-if: true}\n",
                "unsupported rust-cache input shape",
            ),
        )
        workflow = CI_YML.read_text()
        for with_map, error in cases:
            with self.subTest(with_map=with_map):
                step = (
                    "\n      - name: Competing cache writer\n"
                    "        uses: Swatinem/rust-cache@v2\n"
                    + with_map
                )
                mutated = workflow.replace(
                    "\n  ci-fast:",
                    step + "\n  ci-fast:",
                    1,
                )
                with self.assertRaisesRegex(AssertionError, error):
                    _assert_shared_rust_cache_writer_contract(mutated)

    def test_read_only_cache_contract_rejects_a_second_cache_step(self):
        block = _ci_job_block("faithful-observation-phase2-oracle")
        second_steps = (
            (
                "\n      - name: Competing cache writer\n"
                "        uses: Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          shared-key: linux-workspace\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses: Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses: 'Swatinem/rust-cache@v2'\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                '\n      - uses: "Swatinem/rust-cache@v2"\n'
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses : Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses: swatinem/rust-cache@v2.7.8\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses: SWATINEM/RUST-CACHE@master\n"
                "        with:\n"
                "          save-if: true\n"
            ),
        )
        for second_step in second_steps:
            with self.subTest(second_step=second_step):
                with self.assertRaisesRegex(
                    AssertionError,
                    "exactly one Swatinem/rust-cache@v2 step",
                ):
                    _assert_read_only_workspace_cache(block + second_step)

    def test_read_only_cache_contract_rejects_escaped_action_references(self):
        block = _ci_job_block("faithful-observation-phase2-oracle")
        mutated = block + (
            '\n      - uses: "\\x53watinem/rust-cache@v2"\n'
            "        with:\n"
            "          save-if: true\n"
        )
        with self.assertRaisesRegex(
            AssertionError,
            "unsupported quoted workflow action reference",
        ):
            _assert_read_only_workspace_cache(mutated)

    def test_read_only_cache_contract_rejects_yaml_action_references(self):
        block = _ci_job_block("faithful-observation-phase2-oracle")
        for uses in (
            "uses: &cache_action Swatinem/rust-cache@v2",
            "uses: *cache_action",
            "uses: !!str Swatinem/rust-cache@v2",
        ):
            with self.subTest(uses=uses):
                mutated = block + (
                    f"\n      - {uses}\n"
                    "        with:\n"
                    "          save-if: true\n"
                )
                with self.assertRaisesRegex(
                    AssertionError,
                    "unsupported workflow action reference",
                ):
                    _assert_read_only_workspace_cache(mutated)

    def test_read_only_cache_contract_rejects_unsupported_step_maps(self):
        block = _ci_job_block("faithful-observation-phase2-oracle")
        second_steps = (
            (
                "\n      - ? uses\n"
                "        : Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - {uses: Swatinem/rust-cache@v2, "
                "with: {save-if: true}}\n"
            ),
            "\n      - *competing_cache_step\n",
            (
                "\n      - uses: |-\n"
                "          Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          save-if: true\n"
            ),
            (
                "\n      - uses:\n"
                "          Swatinem/rust-cache@v2\n"
                "        with:\n"
                "          save-if: true\n"
            ),
        )
        for second_step in second_steps:
            with self.subTest(second_step=second_step):
                with self.assertRaisesRegex(
                    AssertionError,
                    "unsupported workflow (?:step shape|action reference)",
                ):
                    _assert_read_only_workspace_cache(block + second_step)

    def test_shared_cache_contract_rejects_an_explicit_target_override(self):
        block = _ci_job_block("faithful-observation-phase2-oracle")
        mutated = block.replace(
            "      CARGO_PROFILE_TEST_DEBUG: 0",
            "      CARGO_PROFILE_TEST_DEBUG: 0\n"
            "      CARGO_TARGET_DIR: ${{ github.workspace }}/target",
            1,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "explicit target path changes rust-cache's environment hash",
        ):
            _assert_read_only_workspace_cache(mutated)

    def test_profile_partition_set_math_runs_continuously(self):
        workspace_block = _ci_job_block("full-workspace")
        generalization_block = _ci_job_block("generalize-sweep-oracle-shard")
        default_census = (
            ".venv/bin/python -m unittest "
            "scripts.test_nextest_profile_partition.ProfilePartitionTests"
        )
        generalization_census = (
            ".venv/bin/python -m unittest "
            "scripts.test_nextest_profile_partition.GeneralizationPartitionTests"
        )
        _assert_executable_run_once(workspace_block, default_census)
        _assert_executable_run_once(generalization_block, generalization_census)
        # Each census lists the configuration its job has already compiled.
        # Listing the feature-enabled generalization lane on the workspace
        # shard recompiled the workspace a second time (4.4 hosted minutes).
        self.assertNotIn("GeneralizationPartitionTests", workspace_block)
        self.assertNotIn("ProfilePartitionTests", generalization_block)
        # Both jobs are partitioned, and neither census is part of its job's
        # partition: each lists the whole workspace once. One shard owns it,
        # or four runners repeat the same listing.
        self.assertIn(
            "- name: Verify nextest profile coverage\n"
            "        if: matrix.shard == 1",
            workspace_block,
        )
        self.assertIn(
            "- name: Verify generalization lane selection\n"
            "        if: matrix.shard == 1",
            generalization_block,
        )

    def test_quoted_oracle_name_is_not_an_executable_oracle_step(self):
        block = _ci_job_block("dtype-phase3-oracle")
        oracle_command = "run: .venv/bin/python scripts/dtype_phase3_oracle.py"
        mutated = block.replace(
            oracle_command,
            'run: "true # scripts/dtype_phase3_oracle.py"',
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(
                mutated,
                oracle_command.removeprefix("run: "),
            )

    def test_commented_oracle_plus_noop_is_not_an_executable_oracle_step(self):
        block = _ci_job_block("dtype-phase3-oracle")
        command = ".venv/bin/python scripts/dtype_phase3_oracle.py"
        mutated = block.replace(
            f"run: {command}",
            f"# run: {command}\n        run: \"true\"",
            1,
        )
        with self.assertRaises(AssertionError):
            _assert_executable_run_once(mutated, command)

    def test_non_gate_jobs_are_excluded_by_name(self):
        # Non-gate jobs keep their own commands. Pin the exclusion list so a
        # rename or removal requires an explicit scope update.
        invocations = _parse_ci_run_commands()
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

    def test_every_ci_job_is_scope_classified(self):
        commands = _parse_ci_run_commands()
        gate_workers = set(GATE_WORKER_RUN_COMMANDS)
        unclassified = set(commands) - gate_workers - NON_GATE_JOBS
        self.assertEqual(
            unclassified,
            set(),
            (
                "CI job(s) are neither gate workers nor "
                f"explicit NON_GATE_JOBS: {sorted(unclassified)}"
            ),
        )

    def test_parser_reads_quoted_inline_commands(self):
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                workflow = (
                    "jobs:\n"
                    "  probe:\n"
                    "    steps:\n"
                    f"      - run: {quote}cargo check -p chelis-types{quote}\n"
                )
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["probe"],
                    ["cargo check -p chelis-types"],
                )

    def test_parser_reads_shell_quoted_commands_without_interpreting_them(self):
        for command in (
            '"cargo" check -p chelis-types',
            "c'a'rgo check -p chelis-types",
            r"car\go check -p chelis-types",
            '"chelis" lint --check .',
        ):
            with self.subTest(command=command):
                workflow = (
                    "jobs:\n"
                    "  probe:\n"
                    "    steps:\n"
                    f"      - run: {command}\n"
                )
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["probe"],
                    [command],
                )

    def test_parser_retains_benign_shell_text_without_classifying_it(self):
        workflow = (
            "jobs:\n"
            "  probe:\n"
            "    steps:\n"
            "      - run: echo cargo\n"
        )
        self.assertEqual(
            _parse_ci_run_commands(workflow)["probe"],
            ["echo cargo"],
        )

    def test_parser_marks_literal_and_folded_yaml_run_blocks(self):
        for indicator in ("|", "|-", "|+", ">", ">-", ">+", "|2", ">2-"):
            with self.subTest(indicator=indicator):
                workflow = (
                    "jobs:\n"
                    "  probe:\n"
                    "    steps:\n"
                    f"      - run: {indicator}\n"
                    "          cargo check -p chelis-types\n"
                )
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["probe"],
                    ["<multiline-run-block>"],
                )

    def test_parser_reconstructs_plain_and_quoted_yaml_scalars(self):
        cases = (
            (
                "      - run: 'cargo check -p chelis-types' # comment\n",
                "cargo check -p chelis-types",
            ),
            (
                "      - run:\n"
                "          cargo check -p chelis-types\n",
                "cargo check -p chelis-types",
            ),
            (
                "      - run: cargo\n"
                "          check -p chelis-types\n",
                "cargo check -p chelis-types",
            ),
            (
                "      - run: 'cargo check\n"
                "          -p chelis-types'\n",
                "cargo check -p chelis-types",
            ),
            (
                "      - run : cargo check -p chelis-types\n",
                "cargo check -p chelis-types",
            ),
            (
                "      - run:\n"
                "          python3 scripts/gate.py lint-and-unit;\n"
                "          cargo check -p chelis-types\n",
                "python3 scripts/gate.py lint-and-unit; cargo check -p chelis-types",
            ),
        )
        for run_scalar, expected in cases:
            with self.subTest(run_scalar=run_scalar):
                workflow = "jobs:\n  probe:\n    steps:\n" + run_scalar
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["probe"],
                    [expected],
                )

    def test_parser_accepts_full_github_job_id_grammar(self):
        for spelling in (
            "Unclassified_job",
            "'Unclassified_job'",
            '"Unclassified_job"',
        ):
            with self.subTest(spelling=spelling):
                workflow = (
                    "jobs:\n"
                    f"  {spelling}:\n"
                    "    needs: [changes]\n"
                    "    if: always()\n"
                    "    steps:\n"
                    "      - run: cargo check -p chelis-types\n"
                )
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["Unclassified_job"],
                    ["cargo check -p chelis-types"],
                )
                self.assertEqual(
                    _parse_job_attrs(workflow)["Unclassified_job"],
                    {"needs": "[changes]", "if": "always()"},
                )
                self.assertIn(
                    "Unclassified_job", _workflow_job_blocks(workflow)
                )

    def test_escaped_quoted_job_ids_fail_closed(self):
        workflow = (
            "jobs:\n"
            '  "\\x51uoted_Job":\n'
            "    steps:\n"
            "      - run: cargo check -p chelis-types\n"
        )
        for parser in (
            _parse_ci_run_commands,
            _parse_job_attrs,
            _workflow_job_blocks,
        ):
            with self.subTest(parser=parser.__name__):
                with self.assertRaisesRegex(
                    AssertionError,
                    "unsupported quoted workflow job id",
                ):
                    parser(workflow)

    def test_unsupported_job_mapping_shapes_fail_closed(self):
        workflows = (
            (
                "jobs:\n"
                "  ? Explicit_Job\n"
                "  :\n"
                "    runs-on: ubuntu-latest\n"
            ),
            (
                "jobs:\n"
                "  Hidden_Job: &hidden_job\n"
                "    runs-on: ubuntu-latest\n"
            ),
            "jobs: {Inline_Job: {runs-on: ubuntu-latest}}\n",
        )
        for workflow in workflows:
            for parser in (
                _parse_ci_run_commands,
                _parse_job_attrs,
                _workflow_job_blocks,
            ):
                with self.subTest(workflow=workflow, parser=parser.__name__):
                    with self.assertRaisesRegex(
                        AssertionError,
                        "unsupported workflow job",
                    ):
                        parser(workflow)

    def test_comments_inside_jobs_are_not_job_ids(self):
        """A YAML comment is legal inside `jobs:` and is never a job id.

        Both spellings are here on purpose. The pattern's `\\S.*?` starts at
        `#`, so a comment whose last character is a colon matches the job-id
        regex and used to be reported as a malformed job id, while the same
        comment ending in a period did not (chelis#1443). Only the final
        character differed, so testing one spelling proves nothing about the
        other.
        """
        for trailer in (".", ":"):
            workflow = (
                "jobs:\n"
                f"  # Rule-id: GATE-SCOPE-EXAMPLE -- why this job is exempt{trailer}\n"
                "  exempt-job:\n"
                "    runs-on: ubuntu-latest\n"
                "    steps:\n"
                "      - run: cargo check -p chelis-types\n"
            )
            with self.subTest(trailer=trailer):
                self.assertIsNone(
                    _parse_workflow_job_header(
                        f"  # Rule-id: GATE-SCOPE-EXAMPLE -- why this job is exempt{trailer}"
                    )
                )
                # The comment must not displace the job that follows it.
                self.assertIn("exempt-job", _parse_job_attrs(workflow))
                self.assertIn("exempt-job", _workflow_job_blocks(workflow))
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["exempt-job"],
                    ["cargo check -p chelis-types"],
                )

    def test_unsupported_run_step_shapes_fail_closed(self):
        steps = (
            "      - run: *hidden_command\n",
            "      - run: &hidden_command cargo check -p chelis-types\n",
            '      - run: !!str "\\x63argo check -p chelis-types"\n',
            (
                "      - ? run\n"
                "        : cargo check -p chelis-types\n"
            ),
            "      - {run: cargo check -p chelis-types}\n",
            "      - *hidden_run_step\n",
            '      - "run": cargo check -p chelis-types\n',
        )
        for step in steps:
            workflow = "jobs:\n  probe:\n    steps:\n" + step
            with self.subTest(step=step):
                with self.assertRaisesRegex(
                    AssertionError,
                    "unsupported workflow (?:run scalar|step shape)",
                ):
                    _parse_ci_run_commands(workflow)

    def test_escaped_double_quoted_scalars_fail_closed(self):
        cases = (
            '      - run: "\\x63argo check -p chelis-types"\n',
            '      - run: "car\\\n'
            '          go check -p chelis-types"\n',
        )
        for run_scalar in cases:
            with self.subTest(run_scalar=run_scalar):
                workflow = "jobs:\n  probe:\n    steps:\n" + run_scalar
                self.assertEqual(
                    _parse_ci_run_commands(workflow)["probe"],
                    ["<unsupported-run-scalar>"],
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
        # Multi-line shell bodies are deliberately outside the exact scalar
        # contract for gate-owned workers.
        invocations = _parse_ci_run_commands()
        for job in GATE_WORKER_RUN_COMMANDS:
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

    def test_workflow_reader_preserves_raw_newlines_for_the_digest(self):
        raw = NIX_PACKAGES_YML.read_bytes().replace(b"\n", b"\r\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nix-packages.yml"
            path.write_bytes(raw)
            text = _read_nix_packages_workflow(path)
        self.assertEqual(text.encode("utf-8"), raw)
        with self.assertRaisesRegex(AssertionError, "reviewed native recipe"):
            _assert_nix_intentional_events_only(text)

    def test_workflow_reader_rejects_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "canonical.yml"
            target.write_bytes(NIX_PACKAGES_YML.read_bytes())
            link = root / "nix-packages.yml"
            link.symlink_to(target)
            with self.assertRaisesRegex(AssertionError, "regular file"):
                _read_nix_packages_workflow(link)

    def test_native_nix_workflow_has_both_authoritative_jobs(self):
        self.assertTrue(NIX_PACKAGES_YML.is_file(), "missing Nix package workflow")
        text = _read_nix_packages_workflow()
        required = [
            "name: Nix Packages (x86_64-linux)",
            "runs-on: ubuntu-latest",
            "name: Nix Packages (aarch64-darwin)",
            "runs-on: macos-latest",
        ]
        for marker in required:
            self.assertIn(marker, text)

    def test_each_native_job_runs_the_complete_flake_check_set(self):
        text = _read_nix_packages_workflow()
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
        text = _read_nix_packages_workflow()
        self.assertEqual(text.count("name: Verify the runner system"), 2)
        self.assertIn('assert system == "x86_64-linux", system', text)
        self.assertIn('assert system == "aarch64-darwin", system', text)

    def test_supported_systems_have_exact_native_job_parity(self):
        contracts = (REPO_ROOT / "nix" / "contracts.nix").read_text(encoding="utf-8")
        workflow = _read_nix_packages_workflow()
        _assert_nix_system_job_parity(contracts, workflow)

    def test_supported_system_without_native_job_fails_parity(self):
        contracts = 'supportedSystems = [ "x86_64-linux" "aarch64-linux" ];'
        workflow = "name: Nix Packages (x86_64-linux)"
        with self.assertRaisesRegex(AssertionError, "aarch64-linux"):
            _assert_nix_system_job_parity(contracts, workflow)

    def test_each_native_job_runs_the_nix_contract_suite_with_project_python(self):
        text = _read_nix_packages_workflow()
        self.assertEqual(text.count("uses: astral-sh/setup-uv@v8.1.0"), 2)
        self.assertEqual(text.count("run: uv venv --python 3.11 .venv"), 2)
        self.assertEqual(
            text.count("run: .venv/bin/python scripts/test_nix_flake_contract.py"),
            2,
        )

    def test_each_native_job_uses_the_reviewed_portable_devenv_base(self):
        text = _read_nix_packages_workflow()
        _assert_native_devenv_recipe(text)

    def test_missing_central_devenv_action_fails_the_native_recipe(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace(f"uses: {DEVENV_SETUP_ACTION}", "uses: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "setup-devenv"):
            _assert_native_devenv_recipe(mutated)

    def test_missing_portable_shell_fails_the_native_recipe(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace(f"shell: {PORTABLE_DEVENV_SHELL}", "shell: bash", 1)
        with self.assertRaisesRegex(AssertionError, "devenv-ci"):
            _assert_native_devenv_recipe(mutated)

    def test_late_central_devenv_action_fails_the_native_recipe(self):
        text = _read_nix_packages_workflow()
        setup = f"uses: {DEVENV_SETUP_ACTION}"
        mutated = text.replace(setup, "uses: omitted", 1).replace(
            "run: devenv test --no-tui",
            f"run: devenv test --no-tui\n      - {setup}",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "precede the runner verification"):
            _assert_native_devenv_recipe(mutated)

    def test_workflow_runs_on_intentional_events_only(self):
        text = _read_nix_packages_workflow()
        _assert_nix_intentional_events_only(text)

    def test_automatic_pr_push_and_schedule_triggers_fail_the_event_lock(self):
        text = _read_nix_packages_workflow()
        for trigger in ("pull_request:", "'pull_request':", "push:", "schedule:"):
            with self.subTest(trigger=trigger):
                mutated = text.replace(
                    "  workflow_dispatch:\n",
                    f"  workflow_dispatch:\n  {trigger}\n",
                    1,
                )
                with self.assertRaisesRegex(AssertionError, "only on"):
                    _assert_nix_intentional_events_only(mutated)

    def test_missing_release_trigger_fails_the_event_lock(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace("  release:\n    types: [published]\n", "", 1)
        with self.assertRaisesRegex(AssertionError, "published releases"):
            _assert_nix_intentional_events_only(mutated)

    def test_event_specific_job_gate_fails_the_event_lock(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace(
            "    runs-on: macos-latest\n",
            "    if: github.event_name == 'workflow_dispatch'\n"
            "    runs-on: macos-latest\n",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "reviewed native recipe"):
            _assert_nix_intentional_events_only(mutated)

    def test_policy_oracle_rejects_comment_and_job_gate_evasions(self):
        text = _read_nix_packages_workflow()
        mutations = {
            "comment-only release": (
                text.replace(
                    "  release:\n    types: [published]\n",
                    "  # release:\n  #   types: [published]\n",
                    1,
                ),
                "published releases",
            ),
            "wrong release action hidden by comment": (
                text.replace(
                    "    types: [published]\n",
                    "    types: [created]\n    # types: [published]\n",
                    1,
                ),
                "published releases",
            ),
            "Linux dispatch blocked": (
                text.replace(
                    "    runs-on: ubuntu-latest\n",
                    "    if: github.event.action == 'published'\n"
                    "    runs-on: ubuntu-latest\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "Darwin disabled": (
                text.replace(
                    "    runs-on: macos-latest\n",
                    "    if: false\n    runs-on: macos-latest\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "event-dependent empty matrix": (
                text.replace(
                    "    runs-on: ubuntu-latest\n",
                    "    strategy:\n"
                    "      matrix:\n"
                    "        lane: ${{ github.event_name == 'workflow_dispatch' "
                    "&& fromJSON('[\"run\"]') || fromJSON('[]') }}\n"
                    "    runs-on: ubuntu-latest\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "event-dependent runner": (
                text.replace(
                    "    runs-on: ubuntu-latest\n",
                    "    runs-on: ${{ github.event_name == 'workflow_dispatch' "
                    "&& 'ubuntu-latest' || 'no-such-runner' }}\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "event-gated complete check step": (
                text.replace(
                    "      - name: Run the complete native flake check set\n"
                    "        run: nix flake check --print-build-logs\n",
                    "      - name: Run the complete native flake check set\n"
                    "        if: github.event_name == 'workflow_dispatch'\n"
                    "        run: nix flake check --print-build-logs\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "escaped event-gated complete check step": (
                text.replace(
                    "      - name: Run the complete native flake check set\n"
                    "        run: nix flake check --print-build-logs\n",
                    "      - name: Run the complete native flake check set\n"
                    "        run: nix flake check --print-build-logs\n"
                    "        if: \"${{ gith\\u0075b.event_name == "
                    "'workflow_dispatch' }}\"\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "complete check allowed to fail": (
                text.replace(
                    "      - name: Run the complete native flake check set\n"
                    "        run: nix flake check --print-build-logs\n",
                    "      - name: Run the complete native flake check set\n"
                    "        continue-on-error: true\n"
                    "        run: nix flake check --print-build-logs\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "sandbox false hidden by comment": (
                text.replace(
                    "        sandbox = true\n",
                    "        sandbox = false # sandbox = true\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "workflow working directory override": (
                text.replace(
                    "jobs:\n",
                    "defaults:\n"
                    "  run:\n"
                    "    working-directory: definitely-missing-native-recipe\n\n"
                    "jobs:\n",
                    1,
                ),
                "reviewed native recipe",
            ),
            "workflow concurrency override": (
                text.replace(
                    "  group: nix-packages-${{ github.ref }}\n",
                    "  group: all-nix-runs-share-one-group\n",
                    1,
                ),
                "reviewed native recipe",
            ),
        }
        for name, (mutated, message) in mutations.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssertionError, message):
                    _assert_nix_intentional_events_only(mutated)

    def test_each_job_bounds_runner_resources(self):
        text = _read_nix_packages_workflow()
        _assert_runner_resource_bounds(text)

    def test_unbounded_build_parallelism_fails_the_resource_lock(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace("        max-jobs = 2\n", "", 1)
        with self.assertRaisesRegex(AssertionError, "max-jobs = 2"):
            _assert_runner_resource_bounds(mutated)

    def test_missing_disk_reclaim_fails_the_resource_lock(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace("name: Reclaim runner disk space", "name: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "reclaim runner disk"):
            _assert_runner_resource_bounds(mutated)

    def test_each_job_caches_the_cvc5_toolchain_closure(self):
        text = _read_nix_packages_workflow()
        _assert_cvc5_closure_cache(text)

    def test_missing_cvc5_restore_fails_the_cache_lock(self):
        text = _read_nix_packages_workflow()
        mutated = text.replace("uses: actions/cache/restore@v4", "uses: omitted", 1)
        with self.assertRaisesRegex(AssertionError, "cvc5 closure"):
            _assert_cvc5_closure_cache(mutated)

    def test_direct_devenv_bootstrap_fails_the_native_recipe(self):
        text = _read_nix_packages_workflow()
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
        self.assertIn(
            "cargo test -p chelis-prove --features smt --test integration_solver_isolation",
            block,
        )
        # chelis#1125 PP7: prove_deep_obligations.rs is `#![cfg(feature =
        # "smt")]` and had no runner anywhere, so its assertion that the `.dp`
        # surface proves at `proof_tier = smt` could not fail. The step below
        # is the runner; pinning it here is what keeps a merged smt-gated test
        # from going uninvoked again.
        self.assertIn(
            "cargo test -p chelis-cli --features smt --test prove_deep_obligations",
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
        with self.assertRaisesRegex(AssertionError, "complete serialized Carcara suite"):
            _assert_carcara_full_suite_command(filtered)

        duplicated = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
            "      - name: Accidental parallel Carcara rerun\n"
            "        run: cargo test -p chelis-prove --features carcara",
        )
        with self.assertRaisesRegex(AssertionError, "complete serialized Carcara suite"):
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

        # The narrowing is to non-executing subcommands only: a Clippy step
        # that enables every feature compiles the Carcara lane without running
        # its suite, and must not read as a second run.
        for lint_only in (
            "cargo clippy --workspace --all-targets --all-features -- -D warnings",
            "cargo check -p chelis-prove --features carcara",
        ):
            with self.subTest(lint_only=lint_only):
                _assert_carcara_full_suite_command(
                    text.replace(
                        f"run: {CARCARA_FULL_SUITE_COMMAND}",
                        f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
                        "      - name: Lint-only Carcara compile\n"
                        f"        run: {lint_only}",
                    )
                )

        # ... but an unrecognized subcommand still counts as executing.
        with self.assertRaisesRegex(
            AssertionError, "complete serialized Carcara suite"
        ):
            _assert_carcara_full_suite_command(
                text.replace(
                    f"run: {CARCARA_FULL_SUITE_COMMAND}",
                    f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
                    "      - name: Unknown Carcara subcommand\n"
                    "        run: cargo some-new-runner -p chelis-prove "
                    "--features carcara",
                )
            )

        multiline = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            f"run: {CARCARA_FULL_SUITE_COMMAND}\n"
            "      - name: Multiline Carcara rerun\n"
            "        run: |\n"
            "          cargo test -p chelis-prove --features carcara\n",
        )
        with self.assertRaisesRegex(AssertionError, "complete serialized Carcara suite"):
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
        with self.assertRaisesRegex(AssertionError, "complete serialized Carcara suite"):
            _assert_carcara_full_suite_command(disabled)

        conditional = text.replace(
            f"run: {CARCARA_FULL_SUITE_COMMAND}",
            "if: ${{ always() }}\n"
            f"        run: {CARCARA_FULL_SUITE_COMMAND}",
        )
        with self.assertRaisesRegex(AssertionError, "complete serialized Carcara suite"):
            _assert_carcara_full_suite_command(conditional)

    def test_full_smt_uses_vendored_gmp_family(self):
        text = SMT_FULL_PROVE_YML.read_text()
        _assert_full_smt_system_packages(text)

        for package in SMT_FULL_VENDORED_SYSTEM_PACKAGES:
            for location, mutated in (
                (
                    "primary",
                    text.replace(" libz3-dev", f" {package} libz3-dev"),
                ),
                (
                    "later",
                    text.replace(
                        "ci_apt_get.py gappa",
                        f"ci_apt_get.py gappa {package}",
                    ),
                ),
            ):
                with self.subTest(package=package, location=location):
                    with self.assertRaisesRegex(
                        AssertionError, "locked native dependencies"
                    ):
                        _assert_full_smt_system_packages(mutated)

    def test_prove_feature_graph_uses_vendored_gmp_family(self):
        text = CHELIS_PROVE_TOML.read_text()
        carcara_result = subprocess.run(
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
        all_result = subprocess.run(
            [
                "cargo",
                "tree",
                "-p",
                "chelis-prove",
                "--all-features",
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
        _assert_prove_uses_vendored_gmp_family(
            text, carcara_result.stdout, all_result.stdout
        )
        with self.assertRaisesRegex(AssertionError, "locked GMP/MPFR/MPC sources"):
            _assert_prove_uses_vendored_gmp_family(
                text,
                carcara_result.stdout,
                all_result.stdout + '\ngmp-mpfr-sys feature "use-system-libs"',
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


def _parse_job_attrs(text: str | None = None) -> dict[str, dict[str, str]]:
    """Parse `.github/workflows/ci.yml` and return, per job, its
    top-level `needs:` and `if:` lines (the first occurrence at the
    job's own indent). Line-based to match the existing parser style and
    avoid a PyYAML dependency the CI venv may not carry."""
    if text is None:
        text = CI_YML.read_text()
    lines = text.splitlines()
    current_job: str | None = None
    attrs: dict[str, dict[str, str]] = {}
    attr_line = re.compile(r"^    (needs|if):\s*(.+?)\s*$")
    # Only parse headers inside the `jobs:` block; `on:` triggers like
    # `  push:` share the two-space indent and would otherwise read as
    # jobs.
    in_jobs = False
    for line in lines:
        if _parse_workflow_jobs_header(line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        job_name = _parse_workflow_job_header(line)
        if job_name is not None:
            current_job = job_name
            attrs.setdefault(current_job, {})
            continue
        if current_job is None:
            continue
        am = attr_line.match(line)
        if am is not None:
            key, value = am.group(1), am.group(2)
            attrs[current_job].setdefault(key, value)
    return attrs


TELEMETRY_JOB = "test-telemetry"


def _telemetry_junit_producers(attrs: dict[str, dict[str, str]]) -> tuple[str, ...]:
    """Return the telemetry job's JUnit producers, in `needs:` order.

    `changes` is a gate input, not a producer. Every other entry runs
    nextest and uploads at least one report the telemetry job later
    downloads by exact name.
    """
    raw = attrs.get(TELEMETRY_JOB, {}).get("needs", "")
    match = re.match(r"^\[(?P<body>.*)\]$", raw.strip())
    if match is None:
        raise AssertionError(f"unreadable telemetry needs list: {raw!r}")
    entries = (entry.strip() for entry in match.group("body").split(","))
    producers = tuple(entry for entry in entries if entry and entry != "changes")
    if not producers:
        raise AssertionError("the telemetry job must name its JUnit producers")
    return producers


def _assert_telemetry_skips_without_every_junit(
    attrs: dict[str, dict[str, str]],
) -> None:
    """Lock the producer-success gate on the cross-lane telemetry job.

    Each download names one artifact and carries no fallback, so a
    producer that never started -- a hosted runner the job was never
    assigned -- and a producer that died at its own upload step both
    leave this job failing on `Artifact not found`. That second red
    describes the infrastructure, not the defect, while the required
    aggregate already reports the producer honestly. Requiring every
    producer to have succeeded is safe only because this aggregate is
    not a required status context and asserts no contract of its own;
    required aggregates stay `always()` and fail closed.
    """
    cond = attrs.get(TELEMETRY_JOB, {}).get("if", "")
    for producer in _telemetry_junit_producers(attrs):
        clause = f"needs.{producer}.result == 'success'"
        if clause not in cond:
            raise AssertionError(
                f"the telemetry job must skip unless {producer!r} succeeded: "
                f"missing {clause!r} in {cond!r}"
            )


class IntegrationPlanHandoffTests(unittest.TestCase):
    """The current run's plan must survive Cargo cache restoration."""

    @staticmethod
    def plan_after_restore(block):
        steps = re.split(r"(?m)^      - ", block)[1:]
        restores = [i for i, step in enumerate(steps) if "uses: Swatinem/rust-cache@" in step]
        downloads = [
            i
            for i, step in enumerate(steps)
            if "name: integration-change-plan\n" in step
            and "path: target/integration-change\n" in step
        ]
        executions = [i for i, step in enumerate(steps) if "scripts/ci_change_owned.py run-shard" in step]
        assert len(restores) == len(downloads) == len(executions) == 1, "one restore, current plan download and executor required"
        restore, download, execute = restores[0], downloads[0], executions[0]
        assert restore < download < execute, "download the current plan after restoring target and before execution"
        assert "path: target/integration-change\n" in steps[download], "download must reach the executor plan path"
        assert "--plan target/integration-change/plan.json\n" in steps[execute], "executor must read the downloaded plan"
        assert (
            "if: steps.shard-selection.outputs.has_targets == 'true'"
            in steps[download]
        ), "current plan download must run exactly for nonempty shards"
        assert "continue-on-error:" not in steps[download], "missing current plan must fail the job"

    def test_workers_download_the_current_plan_after_cache_restore(self):
        for job in ("change-owned-shard", "package-expansion-shard"):
            with self.subTest(job=job):
                self.plan_after_restore(_ci_job_block(job))

    def test_plan_survives_cache_deletion_or_stale_overwrite_in_workflow_order(self):
        for job in ("change-owned-shard", "package-expansion-shard"):
            steps = re.split(r"(?m)^      - ", _ci_job_block(job))[1:]
            for restore_mode in ("delete", "stale"):
                with self.subTest(job=job, cache=restore_mode), tempfile.TemporaryDirectory() as tmp:
                    plan = Path(tmp) / "target/integration-change/plan.json"
                    plan.parent.mkdir(parents=True)
                    for step in steps:
                        if "uses: Swatinem/rust-cache@" in step:
                            if restore_mode == "delete":
                                plan.unlink(missing_ok=True)
                            else:
                                plan.write_text("stale cached plan")
                        elif (
                            "name: integration-change-plan\n" in step
                            and "path: target/integration-change\n" in step
                        ):
                            plan.write_text("current run plan")
                        elif "scripts/ci_change_owned.py run-shard" in step:
                            self.assertEqual(plan.read_text(), "current run plan")

    def test_early_missing_wrong_path_and_optional_downloads_fail(self):
        for job in ("change-owned-shard", "package-expansion-shard"):
            block = _ci_job_block(job)
            steps = re.split(r"(?m)(?=^      - )", block)
            download = next(
                step
                for step in steps
                if "name: integration-change-plan\n" in step
                and "path: target/integration-change\n" in step
            )
            early = download + block.replace(download, "", 1)
            for label, mutant in (
                ("early", early),
                ("missing", block.replace(download, "", 1)),
                ("wrong path", block.replace("path: target/integration-change\n", "path: stale-plan\n", 1)),
                (
                    "wrong condition",
                    block.replace(
                        download,
                        download.replace(
                            "if: steps.shard-selection.outputs.has_targets == 'true'",
                            "if: failure()",
                            1,
                        ),
                        1,
                    ),
                ),
                ("optional", block.replace(download, download.replace("        uses:", "        continue-on-error: true\n        uses:", 1), 1)),
            ):
                with self.subTest(job=job, mutation=label):
                    with self.assertRaises(AssertionError):
                        self.plan_after_restore(mutant)


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
        "lint-rust",
        "script-unit",
        "ci-fast",
        "integration-plan",
        "backend-sanitizers",
        "smt-build",
        "smt-build-glibc231",
    }
    # These workers inherit the docs-only disposition through a required
    # upstream job rather than reading `docs_only` directly.
    DEPENDENCY_GATED_JOBS = {"change-owned-shard"}
    # The stable required context aggregates the parallel integration legs,
    # so it needs their results as well as the docs-only classification.
    HEAVY_AGGREGATOR_JOBS = {
        "lint-and-unit",
        "change-owned-report",
        "integration",
    }
    # Best-effort reporting aggregates run after failed dependencies but may
    # skip on cancellation because they are not required status contexts.
    HEAVY_REPORT_JOBS = {"test-telemetry"}
    # Jobs that use the same always-present `changes` job but key on a
    # narrower contract input rather than on the docs-only classification.
    CHANGE_GATED_JOBS = {
        "diagnostic-kind-oracle",
    }
    # Jobs that run for every implementation event, including docs-only PRs.
    # Generic metadata edits never trigger ci.yml.
    IMPLEMENTATION_ALWAYS_RUN_JOBS = {"no-ai-authorship", "docs"}
    # The classifier is always present on actual implementation events.
    ALWAYS_RUN_JOBS = {"changes"}

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

    def test_change_owned_worker_inherits_the_docs_disposition(self):
        attrs = _parse_job_attrs()
        required = attrs["change-owned-shard"]
        self.assertEqual(required.get("needs"), "[changes, integration-plan]")
        self.assertIn(
            "needs.integration-plan.result == 'success'",
            required.get("if", ""),
        )
        self.assertIn("github.event_name != 'push'", required.get("if", ""))
        for job in self.DEPENDENCY_GATED_JOBS:
            self.assertIn("!cancelled()", attrs[job].get("if", ""))

    def test_integration_aggregator_is_fail_closed_and_docs_gated(self):
        attrs = _parse_job_attrs()
        integration = attrs["integration"]
        self.assertEqual(
            integration.get("needs"),
            "[changes, ci-fast, change-owned-report]",
        )
        cond = integration.get("if", "")
        self.assertIn("always()", cond)
        self.assertNotIn("!cancelled()", cond)
        self.assertIn("needs.changes.result != 'success'", cond)
        self.assertIn("needs.changes.outputs.docs_only != 'true'", cond)
        block = _ci_job_block("integration")
        self.assertIn("needs.ci-fast.result", block)
        self.assertIn("needs.change-owned-report.result", block)
        self.assertIn("scripts/ci_require_success.py", block)
        self.assertIn(
            "      - name: Require the fixed main integration suite",
            block,
        )
        main_step, pr_step = block.split(
            "      - name: Require all pull-request integration legs",
            1,
        )
        self.assertIn("if: github.event_name == 'push'", main_step)
        self.assertIn("ci-fast=${{ needs.ci-fast.result }}", main_step)
        self.assertNotIn("change-owned-report=", main_step)
        self.assertIn("if: github.event_name != 'push'", pr_step)
        self.assertIn("ci-fast=${{ needs.ci-fast.result }}", pr_step)
        self.assertIn(
            "change-owned-report=${{ needs.change-owned-report.result }}",
            pr_step,
        )

    def test_change_owned_jobs_have_exact_pr_only_predicates(self):
        attrs = _parse_job_attrs()
        expected = {
            "integration-plan": (
                "${{ !cancelled() && github.event_name != 'push' && "
                "needs.changes.outputs.candidate_preflight == 'success' && "
                "(needs.changes.result != 'success' || "
                "needs.changes.outputs.docs_only != 'true') }}"
            ),
            "change-owned-shard": (
                "${{ !cancelled() && github.event_name != 'push' && "
                "needs.integration-plan.result == 'success' }}"
            ),
            "change-owned-report": (
                "${{ always() && github.event_name != 'push' && "
                "(needs.changes.result != 'success' || "
                "needs.changes.outputs.docs_only != 'true') }}"
            ),
        }
        for job, predicate in expected.items():
            with self.subTest(job=job):
                self.assertEqual(
                    " ".join(attrs[job].get("if", "").split()),
                    " ".join(predicate.split()),
                )

    def test_generalize_sweep_aggregator_is_fail_closed_nightly(self):
        attrs = _parse_job_attrs(CI_YML.with_name("heavy-e2e.yml").read_text())
        aggregate = attrs["generalize-sweep-oracle"]
        self.assertEqual(
            aggregate.get("needs"),
            "[generalize-sweep-oracle-shard]",
        )
        cond = aggregate.get("if", "")
        self.assertIn("always()", cond)
        self.assertNotIn("!cancelled()", cond)
        block = _ci_job_block("generalize-sweep-oracle")
        self.assertIn("needs.generalize-sweep-oracle-shard.result", block)
        self.assertIn("scripts/ci_require_success.py", block)

    def test_every_required_aggregator_runs_after_cancelled_dependencies(self):
        attrs = _parse_job_attrs()
        for job in self.HEAVY_AGGREGATOR_JOBS:
            with self.subTest(job=job):
                cond = attrs[job].get("if", "")
                self.assertIn("always()", cond)
                self.assertNotIn("!cancelled()", cond)

    def test_nonrequired_report_aggregator_skips_on_cancellation(self):
        attrs = _parse_job_attrs()
        for job in self.HEAVY_REPORT_JOBS:
            with self.subTest(job=job):
                cond = attrs[job].get("if", "")
                self.assertIn("!cancelled()", cond)
                self.assertNotIn("always()", cond)

    def test_telemetry_skips_unless_every_junit_producer_succeeded(self):
        _assert_telemetry_skips_without_every_junit(_parse_job_attrs())

    def test_telemetry_producer_list_matches_the_report_aggregate(self):
        attrs = _parse_job_attrs()
        self.assertIn(TELEMETRY_JOB, self.HEAVY_REPORT_JOBS)
        self.assertEqual(
            _telemetry_junit_producers(attrs),
            (
                "ci-fast",
                    ),
        )

    def test_dropping_one_producer_clause_fails_the_telemetry_gate(self):
        attrs = _parse_job_attrs()
        for producer in _telemetry_junit_producers(attrs):
            with self.subTest(producer=producer):
                mutated = {job: dict(values) for job, values in attrs.items()}
                mutated[TELEMETRY_JOB]["if"] = mutated[TELEMETRY_JOB]["if"].replace(
                    f" && needs.{producer}.result == 'success'", "", 1
                )
                with self.assertRaisesRegex(AssertionError, producer):
                    _assert_telemetry_skips_without_every_junit(mutated)

    def test_a_weaker_non_cancelled_producer_clause_fails_the_gate(self):
        attrs = _parse_job_attrs()
        mutated = {job: dict(values) for job, values in attrs.items()}
        mutated[TELEMETRY_JOB]["if"] = mutated[TELEMETRY_JOB]["if"].replace(
            "needs.ci-fast.result == 'success'",
            "needs.ci-fast.result != 'cancelled'",
            1,
        )
        with self.assertRaisesRegex(AssertionError, "ci-fast"):
            _assert_telemetry_skips_without_every_junit(mutated)

    def test_unreadable_telemetry_needs_list_fails_loudly(self):
        with self.assertRaisesRegex(AssertionError, "unreadable telemetry"):
            _telemetry_junit_producers({TELEMETRY_JOB: {"needs": "changes"}})

    def test_a_producerless_telemetry_needs_list_fails_loudly(self):
        with self.assertRaisesRegex(AssertionError, "must name its JUnit"):
            _telemetry_junit_producers({TELEMETRY_JOB: {"needs": "[changes]"}})

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

    def test_implementation_always_run_jobs_are_not_docs_gated(self):
        attrs = _parse_job_attrs()
        for job in self.IMPLEMENTATION_ALWAYS_RUN_JOBS:
            self.assertEqual(attrs[job].get("needs"), "[changes]")
            condition = attrs[job].get("if", "")
            self.assertEqual(condition, "${{ !cancelled() }}")
            self.assertNotIn("docs_only", condition)

    def test_every_job_is_classified(self):
        # Every ci.yml job is either heavy-gated or always-run; a new job
        # forces a deliberate classification (mirrors the workflow-file
        # scope test).
        attrs = _parse_job_attrs()
        classified = (
            self.HEAVY_GATED_JOBS
            | self.DEPENDENCY_GATED_JOBS
            | self.HEAVY_AGGREGATOR_JOBS
            | self.HEAVY_REPORT_JOBS
            | self.CHANGE_GATED_JOBS
            | self.IMPLEMENTATION_ALWAYS_RUN_JOBS
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
        self.assertIn("route through uv", err.getvalue())


if __name__ == "__main__":
    unittest.main()
