"""Lock the Rust build cache policy across every workflow (chelis#3034).

Swatinem/rust-cache derives its key from the prefix-key and shared-key inputs,
the runner OS, the installed rustc versions, the values of environment
variables whose names start with CARGO, CC, CFLAGS, CXX, CMAKE or RUST, and the
Cargo manifests. A consumer hits only when some writer saved under exactly
that key from a ref the consumer can restore. These checks keep that true:

* every cache step names its family through `shared-key` under one prefix,
  and a job that can land on a self-hosted runner skips the step there;
* each family has exactly one writer job, which saves only from main and only
  after a successful build;
* only the workflows in WRITERS may save a cache, each proven main-only from
  its own triggers and checkouts; every other workflow either saves nothing or
  holds `cache-mode: read`, and the families a read-only workflow restores are
  written by ci-cache-warm.yml;
* every job in a family declares the same key inputs and build environment;
* ci-cache-warm.yml writes from main's own code only and repeats only commands
  its family's consumers run.

The checks read what the workflows declare. An environment variable a step
exports through GITHUB_ENV is outside them.
"""

from __future__ import annotations

import copy
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WARM = "ci-cache-warm.yml"
# The only workflows that may save a GitHub cache. Each must prove from its own
# YAML that it saves from main's code only (assert_writers_save_from_main).
WRITERS = (
    WARM,
    "conformance-nightly.yml",
    "ecosystem-drift.yml",
    "macos-nightly.yml",
    # Saves the prebuilt cvc5 fallback; its Rust cache step only restores.
    "smt-full-prove.yml",
)

PREFIX_KEY = "rust-v1-${{ runner.environment }}"
MAIN_ONLY = "${{ github.ref == 'refs/heads/main' }}"
SAVE_IF = (False, "false", MAIN_ONLY, "${{ github.ref == 'refs/heads/main' && matrix.shard == 1 }}")
HOSTED = "runner.environment == 'github-hosted'"
KEY_ENV = re.compile(r"^(CARGO|CC|CFLAGS|CXX|CMAKE|RUST)")
# Variables a Cargo build, a build script or a C toolchain reads.
BUILD_ENV = re.compile(r"^(CARGO|CC|CFLAGS|CXX|CMAKE|RUST|CPPFLAGS|LDFLAGS|LIBRARY_PATH|PKG_CONFIG|AR$|LD$|NM$|RANLIB$)")
FAMILY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:-\$\{\{ matrix\.[a-z_]+ \}\})?$")
HOSTED_LABEL = re.compile(r"^(ubuntu|macos|windows)-[a-z0-9.]+$")
# setup-project-ci installs the toolchain with this action on hosted runners.
SETUP_TOOLCHAIN = "dtolnay/rust-toolchain@stable"
KEY_INPUTS = ("Cargo.lock", "**/Cargo.toml", "rust-toolchain.toml", ".cargo/config.toml")


def load_workflows() -> dict[str, dict]:
    return {
        path.name: yaml.safe_load(path.read_text())
        for path in sorted(WORKFLOWS.glob("*.yml"))
    }


def triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the Actions key `on` as True.
    value = workflow.get("on", workflow.get(True))
    if isinstance(value, str):
        return {value: None}
    if isinstance(value, list):
        return {name: None for name in value}
    return value or {}


def is_cache(step: dict) -> bool:
    return str(step.get("uses", "")).lower().startswith("swatinem/rust-cache@")


def saves(step: dict) -> bool:
    """Whether a step can write a GitHub cache entry."""
    uses = str(step.get("uses", "")).lower()
    if is_cache(step):
        return (step.get("with") or {}).get("save-if", True) not in (False, "false")
    return uses.startswith(("actions/cache@", "actions/cache/save@"))


def read_only(workflow: dict) -> bool:
    """Whether GitHub refuses every save in the workflow."""
    modes = [workflow.get("cache-mode")]
    modes += [job.get("cache-mode", workflow.get("cache-mode")) for job in (workflow.get("jobs") or {}).values()]
    return all(mode in ("read", "none") for mode in modes)


def cache_steps(workflows: dict[str, dict]):
    """Yield (workflow, job id, job, step index, step) for every cache step."""
    for name, workflow in workflows.items():
        for job_id, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                if is_cache(step):
                    yield name, job_id, job, index, step


def hosted_only_job(job: dict) -> bool:
    runs_on = job.get("runs-on")
    return isinstance(runs_on, str) and bool(HOSTED_LABEL.match(runs_on))


def assert_cache_steps_name_their_family(workflows: dict[str, dict]) -> None:
    for name, job_id, job, _, step in cache_steps(workflows):
        where = f"{name}::{job_id}"
        inputs = step.get("with") or {}
        if inputs.get("prefix-key") != PREFIX_KEY:
            raise AssertionError(f"{where}: prefix-key must be {PREFIX_KEY!r}")
        family = inputs.get("shared-key")
        if not isinstance(family, str) or not FAMILY.match(family):
            raise AssertionError(f"{where}: shared-key must name the cache family, found {family!r}")
        for override in ("key", "add-job-id-key", "add-rust-environment-hash-key"):
            if override in inputs:
                raise AssertionError(f"{where}: {override} would split the family's key")
        if "save-if" not in inputs or inputs["save-if"] not in SAVE_IF:
            raise AssertionError(f"{where}: save-if must be false or main-only, found {inputs.get('save-if')!r}")
        if inputs.get("cache-on-failure", False) not in (False, "false"):
            raise AssertionError(
                f"{where}: save only after a successful build; rust-cache never re-saves "
                "a key it restored exactly, so a partial entry would stay live"
            )
        if not hosted_only_job(job):
            condition = str(step.get("if", ""))
            if condition != HOSTED and not condition.startswith(HOSTED + " && "):
                raise AssertionError(
                    f"{where}: a job that can run self-hosted must skip the GitHub "
                    f"cache there; start the step's if with {HOSTED!r}"
                )


def families(workflows: dict[str, dict]) -> dict[str, list[tuple[str, str, bool]]]:
    """Map each family to its (workflow, job id, writes) members."""
    members: dict[str, list[tuple[str, str, bool]]] = {}
    for name, job_id, _, _, step in cache_steps(workflows):
        inputs = step.get("with") or {}
        writes = inputs.get("save-if") not in (False, "false")
        members.setdefault(inputs.get("shared-key"), []).append((name, job_id, writes))
    return members


def assert_one_writer_per_family(workflows: dict[str, dict]) -> None:
    for family, members in families(workflows).items():
        writers = [(name, job_id) for name, job_id, writes in members if writes]
        if len(writers) != 1:
            raise AssertionError(f"{family} must have exactly one writer, found {writers}")
        name, job_id = writers[0]
        restored_by_candidates = any(read_only(workflows[member]) for member, _, _ in members)
        if restored_by_candidates and name != WARM:
            raise AssertionError(f"{family} is restored by read-only CI, so {WARM} must write it, not {name}")


def own_checkout(step: dict) -> bool:
    repository = (step.get("with") or {}).get("repository")
    return repository in (None, "${{ github.repository }}", "Chelis-Lang/chelis")


def assert_writers_save_from_main(workflows: dict[str, dict]) -> None:
    """Fail closed: a workflow saves only if it is a WRITER proven main-only."""
    for name, workflow in workflows.items():
        saving = [
            f"{job_id}: {step.get('name') or step.get('uses')}"
            for job_id, job in (workflow.get("jobs") or {}).items()
            for step in job.get("steps") or []
            if saves(step)
        ]
        if name not in WRITERS:
            # GitHub refuses saves under cache-mode read, but a Rust cache
            # writer outside WRITERS is still a policy error.
            rust_writer = any(
                is_cache(step) and saves(step)
                for job in (workflow.get("jobs") or {}).values()
                for step in job.get("steps") or []
            )
            if rust_writer or (saving and not read_only(workflow)):
                raise AssertionError(
                    f"{name} is not in WRITERS, so it must save nothing or hold "
                    f"cache-mode: read; it saves in {saving}"
                )
            continue
        if not saving:
            raise AssertionError(f"{name} is listed in WRITERS but saves nothing")
        if read_only(workflow):
            raise AssertionError(f"{name} is listed in WRITERS but holds cache-mode read")
        events = triggers(workflow)
        if set(events) - {"push", "schedule", "workflow_dispatch"}:
            raise AssertionError(f"{name} writes caches, so it may run only on push to main, schedule and dispatch: {sorted(events)}")
        if "push" in events:
            push = events["push"] or {}
            if push.get("branches") != ["main"] or set(push) - {"branches", "paths"}:
                raise AssertionError(f"{name} writes caches, so it may push-trigger on main only")
        for job_id, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                if not str(step.get("uses", "")).startswith("actions/checkout@"):
                    continue
                ref = str((step.get("with") or {}).get("ref", ""))
                if own_checkout(step) and ref not in ("", "${{ github.sha }}"):
                    raise AssertionError(f"{name}::{job_id} writes caches but checks out {ref!r} instead of the triggering commit")
                if any(source in ref for source in ("inputs.", "github.event", "github.head_ref", "refs/pull", "pull/")):
                    raise AssertionError(f"{name}::{job_id} writes caches but takes its checkout ref from {ref!r}")


def setup_before(job: dict, index: int) -> list[tuple[str, str]]:
    """Toolchain installs ahead of the cache step, normalized to the action.

    rust-cache hashes each installed toolchain's rustc version, which the
    action ref and its `toolchain` input select; components do not change it.
    """
    installs = []
    for step in (job.get("steps") or [])[:index]:
        uses = str(step.get("uses", ""))
        if uses == "./.github/actions/setup-project-ci":
            installs.append((SETUP_TOOLCHAIN, ""))
        elif uses.startswith("dtolnay/rust-toolchain@"):
            installs.append((uses, str((step.get("with") or {}).get("toolchain", ""))))
    return installs


def build_env(workflow: dict, job: dict) -> dict[str, list[str]]:
    """Every value each build-relevant variable takes anywhere in the job."""
    values: dict[str, set[str]] = {}
    scopes = [workflow.get("env") or {}, job.get("env") or {}]
    scopes += [step.get("env") or {} for step in job.get("steps") or []]
    for scope in scopes:
        for name, value in scope.items():
            if BUILD_ENV.match(name):
                values.setdefault(name, set()).add(str(value))
    return {name: sorted(found) for name, found in values.items()}


def key_inputs(workflow: dict, job: dict, index: int, step: dict) -> dict:
    env = {**(workflow.get("env") or {}), **(job.get("env") or {})}
    inputs = {name: str(value) for name, value in (step.get("with") or {}).items() if name != "save-if"}
    return {
        "env": {name: str(value) for name, value in env.items() if KEY_ENV.match(name)},
        "build environment": build_env(workflow, job),
        "container": job.get("container"),
        "os": "macos" if "macos" in str(job.get("runs-on")) else "linux",
        "toolchain": setup_before(job, index),
        "cache inputs": inputs,
    }


def assert_family_members_share_a_key(workflows: dict[str, dict]) -> None:
    seen: dict[str, tuple[str, dict]] = {}
    for name, job_id, job, index, step in cache_steps(workflows):
        family = (step.get("with") or {}).get("shared-key")
        inputs = key_inputs(workflows[name], job, index, step)
        if family not in seen:
            seen[family] = (f"{name}::{job_id}", inputs)
            continue
        first, expected = seen[family]
        for field, value in inputs.items():
            if value != expected[field]:
                raise AssertionError(
                    f"{family}: {name}::{job_id} {field} {value!r} differs from "
                    f"{first} {expected[field]!r}, so the two compute different keys"
                )


def run_lines(job: dict) -> set[str]:
    lines = set()
    for step in job.get("steps") or []:
        run = step.get("run")
        if isinstance(run, str):
            lines.add(run.strip())
            lines.update(line.strip() for line in run.splitlines())
    return lines


def assert_warm_workflow_writes_from_main(workflows: dict[str, dict]) -> None:
    warm = workflows[WARM]
    events = triggers(warm)
    if set(events) - {"push", "schedule", "workflow_dispatch"}:
        raise AssertionError(f"{WARM} may run only on push, schedule and dispatch: {sorted(events)}")
    push = events.get("push") or {}
    if push.get("branches") != ["main"]:
        raise AssertionError(f"{WARM} must push-trigger on main only")
    missing = [path for path in (*KEY_INPUTS, f".github/workflows/{WARM}") if path not in push.get("paths", [])]
    if missing:
        raise AssertionError(f"{WARM} push paths miss cache-key inputs: {missing}")
    if warm.get("cache-mode") != "write":
        raise AssertionError(f"{WARM} must declare cache-mode: write")
    for job_id, job in warm["jobs"].items():
        if job.get("if") != "github.ref == 'refs/heads/main'":
            raise AssertionError(f"{WARM}::{job_id} must require refs/heads/main")
        if job.get("runs-on") != "ubuntu-latest" or "cache-mode" in job:
            raise AssertionError(f"{WARM}::{job_id} must run hosted under the workflow's cache-mode")
        for step in job.get("steps") or []:
            if str(step.get("uses", "")).startswith("actions/checkout@") and "ref" in (step.get("with") or {}):
                raise AssertionError(f"{WARM}::{job_id} must check out the pushed commit")
        caches = [step for step in job.get("steps") or [] if is_cache(step)]
        if len(caches) != 1 or caches[0]["with"].get("save-if") != MAIN_ONLY:
            raise AssertionError(f"{WARM}::{job_id} must write exactly one family from main")


def assert_warm_commands_come_from_consumers(workflows: dict[str, dict]) -> None:
    consumers: dict[str, list[dict]] = {}
    for name, _, job, _, step in cache_steps(workflows):
        if name != WARM:
            consumers.setdefault(step["with"]["shared-key"], []).append(job)
    for job_id, job in workflows[WARM]["jobs"].items():
        steps = job.get("steps") or []
        cache_index = next(index for index, step in enumerate(steps) if is_cache(step))
        family = steps[cache_index]["with"]["shared-key"]
        if family not in consumers:
            raise AssertionError(f"{WARM}::{job_id} writes {family}, which nothing restores")
        known = set().union(*(run_lines(consumer) for consumer in consumers[family]))
        builds = [step["run"].strip() for step in steps[cache_index + 1:] if isinstance(step.get("run"), str)]
        if not builds:
            raise AssertionError(f"{WARM}::{job_id} builds nothing after restoring {family}")
        for command in builds:
            if command not in known:
                raise AssertionError(f"{WARM}::{job_id} runs {command!r}, which no {family} consumer runs")


class CachePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows = load_workflows()

    def check_all(self, workflows: dict[str, dict]) -> None:
        assert_cache_steps_name_their_family(workflows)
        assert_writers_save_from_main(workflows)
        assert_one_writer_per_family(workflows)
        assert_family_members_share_a_key(workflows)
        assert_warm_workflow_writes_from_main(workflows)
        assert_warm_commands_come_from_consumers(workflows)

    def mutated(self) -> dict[str, dict]:
        return copy.deepcopy(self.workflows)

    def rejects(self, workflows: dict[str, dict], message: str) -> None:
        with self.assertRaisesRegex(AssertionError, message):
            self.check_all(workflows)

    def job_cache(self, workflows: dict[str, dict], name: str, job_id: str) -> dict:
        return next(step for step in workflows[name]["jobs"][job_id]["steps"] if is_cache(step))

    def test_current_workflows_follow_the_policy(self) -> None:
        self.check_all(self.workflows)

    def test_candidate_families_are_written_by_the_warm_workflow(self) -> None:
        members = families(self.workflows)
        for family in ("lint-rust", "linux-workspace", "docs", "smt-smt-build", "smt-glibc231",
                       "backend-sanitizers", "conformance", "python-wheel-smoke", "diagnostic-kind-oracle"):
            with self.subTest(family=family):
                writers = [(name, job_id) for name, job_id, writes in members[family] if writes]
                self.assertEqual([name for name, _ in writers], [WARM])
                self.assertTrue(any(read_only(self.workflows[name]) for name, _, _ in members[family]))

    def test_candidate_workflows_are_read_only(self) -> None:
        for name in ("ci.yml", "conformance.yml", "pr-package-expansion.yml"):
            with self.subTest(workflow=name):
                self.assertTrue(read_only(self.workflows[name]))
                self.assertNotIn(name, WRITERS)

    def test_a_second_writer_is_rejected(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, "macos-nightly.yml", "macos-ownership-ledger")["with"]["save-if"] = MAIN_ONLY
        self.rejects(workflows, "macos-workspace must have exactly one writer")
        workflows = self.mutated()
        self.job_cache(workflows, "heavy-e2e.yml", "full-workspace")["with"]["save-if"] = MAIN_ONLY
        self.rejects(workflows, "heavy-e2e.yml is not in WRITERS")

    def test_a_family_without_a_writer_is_rejected(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, WARM, "docs")["with"]["save-if"] = False
        self.rejects(workflows, "docs must have exactly one writer")

    def test_candidate_workflows_cannot_write(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, WARM, "lint-rust")["with"]["save-if"] = False
        self.job_cache(workflows, "ci.yml", "lint-rust")["with"]["save-if"] = MAIN_ONLY
        self.rejects(workflows, "ci.yml is not in WRITERS")

    def test_writers_save_only_after_success(self) -> None:
        for name, job_id in ((WARM, "docs"), ("macos-nightly.yml", "smt-build-darwin-arm64")):
            with self.subTest(job=f"{name}::{job_id}"):
                workflows = self.mutated()
                self.job_cache(workflows, name, job_id)["with"]["cache-on-failure"] = True
                self.rejects(workflows, "save only after a successful build")

    def test_a_new_dispatch_writer_checking_out_a_pull_request_is_rejected(self) -> None:
        # Review escape M7: a dispatch workflow that checks out a pull request's
        # merge ref and saves a family nothing else writes.
        workflows = self.mutated()
        workflows["escape.yml"] = {
            True: {"workflow_dispatch": {"inputs": {"pr": {"type": "number"}}}},
            "jobs": {"build": {"runs-on": "ubuntu-latest", "steps": [
                {"uses": "actions/checkout@v6", "with": {"ref": "refs/pull/${{ inputs.pr }}/merge"}},
                {"uses": "Swatinem/rust-cache@v2", "with": {
                    "prefix-key": PREFIX_KEY, "shared-key": "escape", "save-if": MAIN_ONLY}},
                {"run": "cargo build"},
            ]}},
        }
        self.rejects(workflows, "escape.yml is not in WRITERS")
        workflows["escape.yml"]["jobs"]["build"]["steps"][1] = {
            "uses": "actions/cache/save@v4", "with": {"path": "target", "key": "escape"}}
        self.rejects(workflows, "escape.yml is not in WRITERS")

    def test_a_writer_checking_out_an_input_ref_is_rejected(self) -> None:
        # Review escape M8: a nightly writer that checks out a dispatch input.
        workflows = self.mutated()
        nightly = workflows["macos-nightly.yml"]
        triggers(nightly)["workflow_dispatch"] = {"inputs": {"ref": {"type": "string"}}}
        checkout = nightly["jobs"]["macos-workspace-shard"]["steps"][0]
        checkout["with"] = {"ref": "${{ inputs.ref }}"}
        self.rejects(workflows, "instead of the triggering commit")
        checkout["with"] = {"repository": "Chelis-Lang/hull", "ref": "${{ github.event.inputs.ref }}"}
        self.rejects(workflows, "takes its checkout ref from")

    def test_writers_must_be_main_only_and_live(self) -> None:
        workflows = self.mutated()
        triggers(workflows["smt-full-prove.yml"])["pull_request"] = None
        self.rejects(workflows, "may run only on push to main, schedule and dispatch")
        workflows = self.mutated()
        workflows["smt-full-prove.yml"]["jobs"]["full-smt-prove"]["steps"] = [
            step for step in workflows["smt-full-prove.yml"]["jobs"]["full-smt-prove"]["steps"]
            if not saves(step)
        ]
        self.rejects(workflows, "listed in WRITERS but saves nothing")

    def test_a_pull_request_writer_is_rejected(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, WARM, "docs")["with"]["save-if"] = True
        self.rejects(workflows, "save-if must be false or main-only")

    def test_a_job_id_or_custom_key_is_rejected(self) -> None:
        for override in ("key", "add-job-id-key", "add-rust-environment-hash-key"):
            with self.subTest(override=override):
                workflows = self.mutated()
                self.job_cache(workflows, "ci.yml", "docs")["with"][override] = "x"
                self.rejects(workflows, "would split the family's key")

    def test_a_missing_or_different_family_name_is_rejected(self) -> None:
        for value in (None, "${{ 'docs' }}", "linux-${{ 'workspace' }}"):
            with self.subTest(value=value):
                workflows = self.mutated()
                inputs = self.job_cache(workflows, "ci.yml", "docs")["with"]
                if value is None:
                    del inputs["shared-key"]
                else:
                    inputs["shared-key"] = value
                self.rejects(workflows, "shared-key must name the cache family")

    def test_a_stale_prefix_is_rejected(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, "heavy-e2e.yml", "module-oracles")["with"]["prefix-key"] = (
            "devenv-${{ hashFiles('devenv.lock') }}"
        )
        self.rejects(workflows, "prefix-key must be")

    def test_self_hosted_capable_jobs_must_skip_the_cache(self) -> None:
        for name, job_id in (("ci.yml", "lint-rust"), ("heavy-e2e.yml", "script-nightly"), ("conformance.yml", "conformance")):
            with self.subTest(job=f"{name}::{job_id}"):
                workflows = self.mutated()
                self.job_cache(workflows, name, job_id).pop("if")
                self.rejects(workflows, "can run self-hosted")

    def test_a_key_environment_difference_is_rejected(self) -> None:
        workflows = self.mutated()
        workflows["pr-package-expansion.yml"]["jobs"]["package-expansion-shard"]["env"]["CC"] = "clang"
        self.rejects(workflows, "linux-workspace: .* env .* compute different keys")

    def test_a_build_environment_difference_is_rejected(self) -> None:
        # Review finding P2-3: a step-scoped compiler keeps the key but makes
        # every cc-rs build script rerun on a hit.
        for variable, value in (("CC", "clang"), ("CXX", "clang++"), ("RUSTFLAGS", "-Ctarget-cpu=native"), ("CFLAGS", "-O3")):
            with self.subTest(variable=variable):
                workflows = self.mutated()
                steps = workflows["pr-package-expansion.yml"]["jobs"]["package-expansion-shard"]["steps"]
                executor = next(step for step in steps if "run-shard" in str(step.get("run", "")))
                executor.setdefault("env", {})[variable] = value
                self.rejects(workflows, "linux-workspace: .* build environment")

    def test_a_cache_input_difference_is_rejected(self) -> None:
        for name, value in (("cache-bin", False), ("cache-directories", "~/.cache/extra"), ("cache-all-crates", True)):
            with self.subTest(input=name):
                workflows = self.mutated()
                self.job_cache(workflows, "heavy-e2e.yml", "module-oracles")["with"][name] = value
                self.rejects(workflows, "linux-workspace: .* cache inputs")

    def test_a_toolchain_difference_is_rejected(self) -> None:
        workflows = self.mutated()
        steps = workflows["smt-full-prove.yml"]["jobs"]["full-smt-prove"]["steps"]
        install = next(step for step in steps if str(step.get("uses", "")).startswith("dtolnay/rust-toolchain@"))
        install["uses"] = "dtolnay/rust-toolchain@1.98.0"
        self.rejects(workflows, "smt-smt-build: .* toolchain")
        workflows = self.mutated()
        steps = workflows["smt-full-prove.yml"]["jobs"]["full-smt-prove"]["steps"]
        install = next(step for step in steps if str(step.get("uses", "")).startswith("dtolnay/rust-toolchain@"))
        install.setdefault("with", {})["toolchain"] = "nightly"
        self.rejects(workflows, "smt-smt-build: .* toolchain")

    def test_a_target_directory_difference_is_rejected(self) -> None:
        workflows = self.mutated()
        self.job_cache(workflows, "ci.yml", "script-unit")["with"]["workspaces"] = ". -> target"
        self.rejects(workflows, "python-wheel-smoke: .* cache inputs")

    def test_warm_triggers_are_restricted_to_main(self) -> None:
        for event in ("pull_request", "pull_request_target", "workflow_run"):
            with self.subTest(event=event):
                workflows = self.mutated()
                triggers(workflows[WARM])[event] = None
                self.rejects(workflows, "may run only on push")
        workflows = self.mutated()
        triggers(workflows[WARM])["push"]["branches"] = ["**"]
        self.rejects(workflows, "push-trigger on main only")

    def test_warm_push_filter_names_the_key_inputs(self) -> None:
        workflows = self.mutated()
        triggers(workflows[WARM])["push"]["paths"].remove("Cargo.lock")
        self.rejects(workflows, "push paths miss cache-key inputs")

    def test_warm_jobs_require_main_and_the_pushed_commit(self) -> None:
        workflows = self.mutated()
        workflows[WARM]["jobs"]["docs"]["if"] = "always()"
        self.rejects(workflows, "must require refs/heads/main")
        workflows = self.mutated()
        checkout = workflows[WARM]["jobs"]["docs"]["steps"][0]
        checkout["with"] = {"ref": "${{ github.event.pull_request.head.sha }}"}
        self.rejects(workflows, "instead of the triggering commit")
        workflows = self.mutated()
        workflows[WARM]["cache-mode"] = "read"
        self.rejects(workflows, "listed in WRITERS but holds cache-mode read")
        workflows = self.mutated()
        del workflows[WARM]["cache-mode"]
        self.rejects(workflows, "must declare cache-mode: write")

    def test_warm_commands_must_come_from_a_consumer(self) -> None:
        workflows = self.mutated()
        workflows[WARM]["jobs"]["docs"]["steps"].append({"run": "cargo build --workspace --all-features"})
        self.rejects(workflows, "which no docs consumer runs")


if __name__ == "__main__":
    unittest.main()
