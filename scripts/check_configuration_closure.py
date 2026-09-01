#!/usr/bin/env python3
"""Prove the workspace's build-configuration space is declared, covered, and complete.

This is the completeness leg of the hash-order determinism contract
(``spec/design/hash_order_determinism.md`` C2.3). The ban on
``std::collections::HashMap``/``HashSet`` is enforced by Clippy's
``disallowed_types`` on real HIR, which sees through aliases, glob imports,
type-alias definition sites, macro expansion, and ``include!``-ed generated
code. Clippy only sees the configuration it compiles, so the question this
script answers is not "which files might rustc read" but the two questions
that actually bound the ban:

1. **Is the configuration space finite and declared?** ``unexpected_cfgs`` is
   denied workspace-wide, so a ``#[cfg(name)]`` whose name is neither a
   declared Cargo feature nor a well-known rustc cfg is a compile error. The
   space is therefore exactly *declared features x supported hosts*.
2. **Does the Clippy matrix cover it, and did it actually compile every
   source?** Leg 2 checks every declared feature is named by a registered
   Clippy invocation whose owner really runs it. Leg 3 reconciles the
   repository's ``.rs`` files against *rustc's own dep-info*, so the compiler
   reports what it compiled instead of a script recomputing it beside the
   compiler.

Leg 3 is the structural point. A previous revision of this contract computed
the compiled-source set from source text, and needed a Rust lexer, a
``#[path]`` resolver, a module-graph walk, doc-comment desugaring, and macro
token-tree inspection to do it - each added after a review found the previous
one incomplete. The compiler already knows the answer and writes it to
``target/<profile>/deps/*.d``; ask it.

Acceptance is exit 0 with the final line ``CONFIGURATION CLOSURE: PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationClosureFailure(RuntimeError):
    """A configuration-space or source-coverage obligation is unmet."""


@dataclass(frozen=True)
class ClippyRun:
    """One registered Clippy invocation, what it compiles, and who runs it."""

    label: str
    command: tuple[str, ...]
    owner: str
    hosts: tuple[str, ...]
    cadence: str

    @property
    def covers_all_features(self) -> bool:
        return "--all-features" in self.command

    def explicit_features(self) -> frozenset[tuple[str, str]]:
        """The `(package, feature)` pairs this command names explicitly."""
        pairs: set[tuple[str, str]] = set()
        for index, argument in enumerate(self.command):
            if argument != "--features" or index + 1 >= len(self.command):
                continue
            for entry in self.command[index + 1].replace(",", " ").split():
                package, separator, feature = entry.partition("/")
                if not separator:
                    raise ConfigurationClosureFailure(
                        f"Clippy run {self.label!r} names the unqualified feature "
                        f"{entry!r}; use `package/feature` so coverage is attributable"
                    )
                pairs.add((package, feature))
        return frozenset(pairs)


@dataclass(frozen=True)
class UncompiledException:
    """A repository directory deliberately outside the workspace build."""

    directory: str
    reason: str
    owning_gate: str


PER_PULL_REQUEST = "per-pull-request"
NIGHTLY = "nightly"

# The registered matrix. Every declared workspace feature must be reachable
# from one of these, and each command must appear verbatim in its owner file.
#
# Cadence is recorded rather than assumed: `z3`, `carcara`, and `arb` need
# external solver toolchains whose per-pull-request cost the repository has
# already declined (see .github/workflows/smt-full-prove.yml's header), so
# their Clippy coverage is nightly. That residual is bounded and named; it is
# not a claim of per-pull-request coverage.
CLIPPY_MATRIX: tuple[ClippyRun, ...] = (
    ClippyRun(
        label="default-features",
        command=(
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ),
        owner="scripts/gate.py",
        hosts=("linux", "macos"),
        cadence=PER_PULL_REQUEST,
    ),
    ClippyRun(
        label="solver-free-features",
        command=(
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--features",
            "chelis-backend-c/sleef,"
            "chelis-e2e/hip-local-gpu,"
            "chelis-prove/clarabel,"
            "chelis-python/extension-module,"
            "chelis-runtime/ownership-ledger,"
            "chelis-types/checkpoint-compile-probe,"
            "chelis-types/generalize-sweep-oracle,"
            "chelis-types/hash-order-compile-probe",
            "--",
            "-D",
            "warnings",
        ),
        owner="scripts/gate.py",
        hosts=("linux", "macos"),
        cadence=PER_PULL_REQUEST,
    ),
    ClippyRun(
        label="cvc5-features",
        command=(
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--features",
            "chelis-cli/smt,chelis-prove/smt,chelis-tide/smt",
            "--",
            "-D",
            "warnings",
        ),
        owner=".github/workflows/ci.yml",
        hosts=("linux",),
        cadence=PER_PULL_REQUEST,
    ),
    ClippyRun(
        label="all-features",
        command=(
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ),
        owner=".github/workflows/smt-full-prove.yml",
        hosts=("linux",),
        cadence=NIGHTLY,
    ),
)

@dataclass(frozen=True)
class NightlyOnlySource:
    """A source whose only Clippy coverage is a nightly matrix row."""

    path: str
    feature: str
    row: str


# The exact residual. These sources sit behind solver features whose external
# toolchains the repository does not provision per pull request, so the only
# row that compiles them is nightly. The list is self-pruning: a file that a
# per-pull-request row does compile is reported as stale, and `--require-complete`
# (run by the nightly job) drops the allowance entirely, so a new uncovered file
# cannot hide behind it.
NIGHTLY_ONLY_SOURCES: tuple[NightlyOnlySource, ...] = (
    NightlyOnlySource(
        path="crates/chelis-prove/src/z3_engine.rs",
        feature="chelis-prove/z3",
        row="all-features",
    ),
    NightlyOnlySource(
        path="crates/chelis-prove/src/bin/certify_erf_envelope.rs",
        feature="chelis-prove/arb",
        row="all-features",
    ),
    NightlyOnlySource(
        path="crates/chelis-prove/src/bin/certify_special_fn_envelope.rs",
        feature="chelis-prove/arb",
        row="all-features",
    ),
)

# Directories holding `.rs` sources that the workspace build deliberately does
# not compile. Each is a standalone Cargo project driven by its own named gate,
# which compiles it and asserts the expected rejection.
UNCOMPILED_EXCEPTIONS: tuple[UncompiledException, ...] = (
    UncompiledException(
        directory="crates/chelis-unord/tests/compile_fail",
        reason="standalone rejection fixtures; compiling them is the test",
        owning_gate="scripts/check_hash_order_phase_b_compile_fail.py",
    ),
    UncompiledException(
        directory="crates/chelis-types/tests/compile_fail",
        reason="standalone rejection fixtures; compiling them is the test",
        owning_gate="scripts/check_checkpoint_compile_fail.py",
    ),
    UncompiledException(
        directory="crates/chelis-compiler-api/tests/compile_fail",
        reason="standalone rejection fixtures; compiling them is the test",
        owning_gate="scripts/check_pipeline_core_compile_fail.py",
    ),
)


def _run_git(repo_root: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def workspace_members(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    manifest = tomllib.loads((repo_root / "Cargo.toml").read_text(encoding="utf-8"))
    return tuple(manifest["workspace"]["members"])


def check_declared_configuration_space(repo_root: Path = REPO_ROOT) -> None:
    """Leg 1: an undeclared `cfg` name must be a compile error everywhere."""
    manifest = tomllib.loads((repo_root / "Cargo.toml").read_text(encoding="utf-8"))
    workspace_level = (
        manifest.get("workspace", {})
        .get("lints", {})
        .get("rust", {})
        .get("unexpected_cfgs")
    )
    if workspace_level != "deny":
        raise ConfigurationClosureFailure(
            'Cargo.toml [workspace.lints.rust] must set unexpected_cfgs = "deny"; '
            f"found {workspace_level!r}. Without it an undeclared cfg compiles and "
            "hides its items from every Clippy run that does not pass that flag."
        )

    uninherited: list[str] = []
    for member in workspace_members(repo_root):
        member_manifest = tomllib.loads(
            (repo_root / member / "Cargo.toml").read_text(encoding="utf-8")
        )
        lints = member_manifest.get("lints", {})
        if lints.get("workspace") is True:
            continue
        if lints.get("rust", {}).get("unexpected_cfgs") == "deny":
            continue
        uninherited.append(member)
    if uninherited:
        raise ConfigurationClosureFailure(
            "these workspace members do not deny unexpected_cfgs: "
            + ", ".join(sorted(uninherited))
            + ". Add `[lints]` with `workspace = true` to each manifest, or set "
            '`[lints.rust] unexpected_cfgs = "deny"` beside its existing lint table.'
        )


def declared_features(repo_root: Path = REPO_ROOT) -> dict[str, frozenset[str]]:
    """Every non-default feature declared by every member, keyed by package name."""
    features: dict[str, frozenset[str]] = {}
    for member in workspace_members(repo_root):
        manifest = tomllib.loads(
            (repo_root / member / "Cargo.toml").read_text(encoding="utf-8")
        )
        declared = {name for name in manifest.get("features", {}) if name != "default"}
        if declared:
            features[manifest["package"]["name"]] = frozenset(declared)
    return features


def check_matrix_covers_declared_features(
    repo_root: Path = REPO_ROOT,
    matrix: Sequence[ClippyRun] = CLIPPY_MATRIX,
) -> None:
    """Leg 2: every declared feature is compiled by some registered Clippy run."""
    if not matrix:
        raise ConfigurationClosureFailure("the Clippy matrix is empty")

    covered: set[tuple[str, str]] = set()
    covers_everything = False
    for run in matrix:
        if run.covers_all_features:
            covers_everything = True
        covered |= run.explicit_features()

    declared = {
        (package, feature)
        for package, features in declared_features(repo_root).items()
        for feature in features
    }

    if not covers_everything:
        uncovered = sorted(
            f"{package}/{feature}" for package, feature in declared - covered
        )
        if uncovered:
            raise ConfigurationClosureFailure(
                "these declared features are compiled by no registered Clippy run, so "
                "a raw hash collection behind them would not be linted: "
                + ", ".join(uncovered)
                + ". Add the feature to an existing CLIPPY_MATRIX command, or register "
                "a new run with the file that owns it."
            )

    stale = sorted(f"{package}/{feature}" for package, feature in covered - declared)
    if stale:
        raise ConfigurationClosureFailure(
            "CLIPPY_MATRIX names features that no workspace member declares: "
            + ", ".join(stale)
            + ". A renamed or removed feature must leave the matrix in the same change."
        )

    for run in matrix:
        check_owner_invokes(run, repo_root)


def gate_command_lines(repo_root: Path = REPO_ROOT) -> frozenset[str]:
    """The exact command lines `scripts/gate.py` publishes through `--list`.

    Reading the gate's own rendering, rather than grepping its source, means a
    command spelled across adjacent Python string literals still matches, and a
    constant that is defined but never reaches a stage does not.
    """
    completed = subprocess.run(
        (sys.executable, "scripts/gate.py", "--list"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = set()
    for line in completed.stdout.splitlines():
        rendered = line.split("  #", 1)[0].strip()
        if rendered:
            lines.add(rendered)
    return frozenset(lines)


def check_owner_invokes(run: ClippyRun, repo_root: Path = REPO_ROOT) -> None:
    """The file a run names must really issue that exact command."""
    owner = repo_root / run.owner
    if not owner.is_file():
        raise ConfigurationClosureFailure(
            f"Clippy run {run.label!r} names a missing owner: {run.owner}"
        )
    shell_form = " ".join(run.command)
    if run.owner == "scripts/gate.py":
        if shell_form in gate_command_lines(repo_root):
            return
    elif shell_form in owner.read_text(encoding="utf-8"):
        return
    raise ConfigurationClosureFailure(
        f"Clippy run {run.label!r} is not invoked by its owner {run.owner}. "
        f"Expected to find: {shell_form}"
    )


def parse_dep_info(path: Path) -> set[str]:
    """Return the prerequisite paths rustc recorded in one dep-info file."""
    prerequisites: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith(" "):
            continue
        separator = line.find(": ")
        if separator == -1:
            continue
        remainder = line[separator + 2 :]
        current = ""
        index = 0
        while index < len(remainder):
            character = remainder[index]
            if character == "\\" and index + 1 < len(remainder):
                current += remainder[index + 1]
                index += 2
            elif character == " ":
                if current:
                    prerequisites.add(current)
                current = ""
                index += 1
            else:
                current += character
                index += 1
        if current:
            prerequisites.add(current)
    return prerequisites


def compiled_rust_sources(
    target_directories: Iterable[Path],
    repo_root: Path = REPO_ROOT,
) -> set[str]:
    """The repository-relative `.rs` files rustc reports having read.

    Stale dep-info can only name a file some earlier run of this matrix did
    compile, so it cannot mask a file the matrix has never compiled, which is
    the property this leg asserts.
    """
    repo_root = repo_root.resolve()
    compiled: set[str] = set()
    for target_directory in target_directories:
        if not target_directory.is_dir():
            continue
        for dep_info in target_directory.rglob("*.d"):
            for prerequisite in parse_dep_info(dep_info):
                if not prerequisite.endswith(".rs"):
                    continue
                candidate = Path(prerequisite)
                if not candidate.is_absolute():
                    candidate = repo_root / candidate
                try:
                    relative = candidate.resolve().relative_to(repo_root)
                except (ValueError, OSError):
                    continue
                compiled.add(relative.as_posix())
    return compiled


def repository_rust_sources(repo_root: Path = REPO_ROOT) -> set[str]:
    """Every `.rs` file a developer can see: tracked plus untracked-not-ignored."""
    return set(
        _run_git(
            repo_root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.rs",
        )
    )


def check_every_source_is_compiled(
    target_directories: Sequence[Path],
    repo_root: Path = REPO_ROOT,
    exceptions: Sequence[UncompiledException] = UNCOMPILED_EXCEPTIONS,
    nightly_only: Sequence[NightlyOnlySource] = NIGHTLY_ONLY_SOURCES,
    require_complete: bool = False,
) -> None:
    """Leg 3: reconcile repository sources against rustc's own dep-info."""
    labels = {run.label: run for run in CLIPPY_MATRIX}
    for source in nightly_only:
        if not (repo_root / source.path).is_file():
            raise ConfigurationClosureFailure(
                f"stale nightly-only source: {source.path} does not exist"
            )
        run = labels.get(source.row)
        if run is None or run.cadence != NIGHTLY:
            raise ConfigurationClosureFailure(
                f"nightly-only source {source.path} names {source.row!r}, which is "
                "not a nightly row of CLIPPY_MATRIX"
            )
    for exception in exceptions:
        if not (repo_root / exception.directory).is_dir():
            raise ConfigurationClosureFailure(
                f"stale uncompiled-source exception: {exception.directory} does not exist"
            )
        if not (repo_root / exception.owning_gate).is_file():
            raise ConfigurationClosureFailure(
                f"uncompiled-source exception {exception.directory} names a missing "
                f"owning gate: {exception.owning_gate}"
            )

    compiled = compiled_rust_sources(target_directories, repo_root)
    if not compiled:
        raise ConfigurationClosureFailure(
            "no dep-info was found, so nothing was reconciled. Run the registered "
            "Clippy matrix first; searched: "
            + ", ".join(str(directory) for directory in target_directories)
        )

    allowed_nightly = (
        set() if require_complete else {source.path for source in nightly_only}
    )
    if not require_complete:
        stale = sorted(path for path in allowed_nightly if path in compiled)
        if stale:
            raise ConfigurationClosureFailure(
                "these sources are recorded as nightly-only but a registered run "
                "compiled them here: "
                + ", ".join(stale)
                + ". Delete their NIGHTLY_ONLY_SOURCES entries; the residual has shrunk."
            )

    excepted = tuple(f"{exception.directory}/" for exception in exceptions)
    uncompiled = sorted(
        source
        for source in repository_rust_sources(repo_root) - compiled - allowed_nightly
        if not source.startswith(excepted)
    )
    if uncompiled:
        raise ConfigurationClosureFailure(
            "these repository Rust sources were compiled by no registered Clippy "
            "configuration, so the hash-collection ban is unproven for them: "
            + ", ".join(uncompiled)
            + ". Either add the configuration that compiles them to CLIPPY_MATRIX, "
            "record them in NIGHTLY_ONLY_SOURCES with the nightly row that does, or "
            "record them in UNCOMPILED_EXCEPTIONS with the gate that owns them."
        )


def default_target_directories(repo_root: Path = REPO_ROOT) -> tuple[Path, ...]:
    target = Path(os.environ.get("CARGO_TARGET_DIR") or str(repo_root / "target"))
    profiles = tuple(
        directory
        for directory in (target / "debug", target / "release")
        if directory.is_dir()
    )
    return profiles or (target,)


def validate(
    repo_root: Path = REPO_ROOT,
    target_directories: Sequence[Path] | None = None,
    require_complete: bool = False,
) -> None:
    check_declared_configuration_space(repo_root)
    check_matrix_covers_declared_features(repo_root)
    check_every_source_is_compiled(
        target_directories
        if target_directories is not None
        else default_target_directories(repo_root),
        repo_root,
        require_complete=require_complete,
    )
    print("CONFIGURATION CLOSURE: PASS", flush=True)


def main(argv: Sequence[str]) -> int:
    directories: list[Path] = []
    require_complete = False
    for argument in argv:
        if argument == "--require-complete":
            require_complete = True
            continue
        if argument.startswith("-"):
            print(
                "usage: check_configuration_closure.py [--require-complete] "
                "[TARGET_DIR ...]",
                file=sys.stderr,
            )
            return 2
        directories.append(Path(argument))
    try:
        validate(
            target_directories=directories or None,
            require_complete=require_complete,
        )
    except (ConfigurationClosureFailure, OSError, subprocess.SubprocessError) as error:
        print(f"CONFIGURATION CLOSURE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
