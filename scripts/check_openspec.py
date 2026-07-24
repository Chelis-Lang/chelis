"""OpenSpec governance checker (Chelis-owned; stdlib-only, Python 3.11+).

Owning change: `openspec/changes/adopt-openspec-governance`. The shared
`Chelis-Lang/ci/actions/openspec-governance` action invokes exactly:

    python3 scripts/check_openspec.py --self-test --merge-bound --base <base>

Local commands and expected success conditions:

    python3 scripts/check_openspec.py --self-test
        exit 0 iff every planted-negative control still rejects.
    python3 scripts/check_openspec.py --pre-archive --base origin/main
        exit 0 iff the branch carries one valid ACTIVE lifecycle (or one
        exemption) with ordering, scope, shape, and strict validation green.
    python3 scripts/check_openspec.py --merge-bound --base origin/main
        exit 0 iff no active lifecycle remains: one synchronized, archived
        lifecycle (or one exemption) with every control green.

`GIT_BIN` and `OPENSPEC_BIN` select the executables when set; OpenSpec must
report exactly 1.6.0. Every policy or tool failure exits nonzero; there are
no fallbacks. The unit suite is `scripts/test_check_openspec.py`; the Phase 0
adoption oracle is `scripts/test_openspec_adoption.py` plus the hosted PR
gate.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    sys.exit("check_openspec: Python 3.11 or newer is required")

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

EXPECTED_OPENSPEC_VERSION = "1.6.0"
MAX_EVENT_BYTES = 1024 * 1024
CITATION_PREFIX = "OpenSpec-Change: "
CHANGES_PREFIX = "openspec/changes/"
ARCHIVE_PREFIX = "openspec/changes/archive/"
EXEMPTIONS_PREFIX = "openspec/exemptions/"
GOVERNANCE_PREFIX = "openspec/"
SPEC_PREFIX = "spec/"
MARKER_NAME = ".openspec.yaml"
PLANNING_NEUTRAL = frozenset({".gitignore", ".gitattributes"})
DELTA_OPERATIONS = ("ADDED", "MODIFIED", "REMOVED", "RENAMED")
ARCHIVE_NAME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)$")
EXEMPTION_NAME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)\.toml$"
)
CHECKBOX = re.compile(r"^- \[([ x])\] *(.*)$")

SELF_TEST_CONTROLS = (
    "malformed-spec",
    "missing-negative-scenario",
    "artifact-drift",
    "planning-order-collapse",
    "citation-mismatch",
    "branch-scope-ambiguity",
    "invalid-exemption",
    "symlink-rejection",
    "unchecked-task",
    "active-merge-state",
    "malformed-archive",
    "unsynchronized-delta",
)


class CheckError(RuntimeError):
    pass


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_openspec", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    parser.add_argument("--merge-bound", action="store_true", dest="merge_bound")
    parser.add_argument("--pre-archive", action="store_true", dest="pre_archive")
    parser.add_argument("--base", dest="base")
    ns = parser.parse_args(argv)
    if ns.merge_bound and ns.pre_archive:
        parser.error("--merge-bound and --pre-archive are mutually exclusive")
    if ns.merge_bound and not ns.base:
        parser.error("--merge-bound requires --base")
    if ns.pre_archive and not ns.base:
        parser.error("--pre-archive requires --base")
    if not (ns.self_test or ns.merge_bound or ns.pre_archive):
        parser.error("select --self-test, --merge-bound, or --pre-archive")
    return ns


def _require_executable(path: Path, label: str) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CheckError(f"{label} executable is missing or not executable: {path}")
    return path


def resolve_git(environ: dict[str, str]) -> Path:
    injected = environ.get("GIT_BIN", "")
    if injected:
        return _require_executable(Path(injected), "injected Git")
    found = shutil.which("git", path=environ.get("PATH", os.defpath))
    if found is None:
        raise CheckError("Git executable is missing")
    return Path(found)


def resolve_openspec(environ: dict[str, str]) -> Path:
    injected = environ.get("OPENSPEC_BIN", "")
    if injected:
        return _require_executable(Path(injected), "injected OpenSpec")
    found = shutil.which("openspec", path=environ.get("PATH", os.defpath))
    if found is None:
        raise CheckError("OpenSpec executable is missing")
    return Path(found)


def verify_openspec(executable, runner=subprocess.run) -> list[str]:
    try:
        completed = runner(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"cannot execute OpenSpec: {error}"]
    actual = completed.stdout.strip()
    if completed.returncode != 0 or actual != EXPECTED_OPENSPEC_VERSION:
        return [
            "OpenSpec version mismatch: expected exactly "
            f"{EXPECTED_OPENSPEC_VERSION}, observed {actual!r} "
            f"(exit {completed.returncode})"
        ]
    return []


def resolve_merge_base(git, base: str, runner=subprocess.run) -> str:
    completed = runner(
        [str(git), "merge-base", base, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CheckError(
            f"comparison base is unresolvable: git merge-base {base} HEAD failed"
        )
    return completed.stdout.strip()


def load_event(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CheckError("event payload must be a regular non-symlink file")
    if path.stat().st_size > MAX_EVENT_BYTES:
        raise CheckError("event payload exceeds the bounded size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckError("event payload is not a valid UTF-8 JSON object") from error
    if not isinstance(payload, dict):
        raise CheckError("event payload root must be an object")
    return payload


def extract_citations(body: str | None) -> list[str]:
    if not body:
        return []
    citations = []
    for line in body.splitlines():
        if line.startswith(CITATION_PREFIX):
            citations.append(line[len(CITATION_PREFIX) :].strip())
    return citations


def check_citation(citations: list[str], lifecycle_id: str) -> list[str]:
    unique = set(citations)
    if not unique:
        return [f"pull request must cite '{CITATION_PREFIX}{lifecycle_id}'"]
    if unique != {lifecycle_id}:
        return [
            f"pull request citation mismatch: cited {sorted(unique)}, "
            f"branch lifecycle is '{lifecycle_id}'"
        ]
    return []


@dataclass(frozen=True)
class Classification:
    active_lifecycles: frozenset[str]
    archives: frozenset[str]
    exemptions: frozenset[str]
    governance: frozenset[str]
    spec: frozenset[str]
    production: frozenset[str]
    hidden: frozenset[str]


def _hidden_component(path: str) -> bool:
    parts = path.split("/")
    for index, part in enumerate(parts):
        if not part.startswith("."):
            continue
        if index == len(parts) - 1 and part == MARKER_NAME:
            continue
        return True
    return False


def classify_paths(paths) -> Classification:
    active: set[str] = set()
    archives: set[str] = set()
    exemptions: set[str] = set()
    governance: set[str] = set()
    spec: set[str] = set()
    production: set[str] = set()
    hidden: set[str] = set()
    for path in paths:
        if path.startswith(GOVERNANCE_PREFIX):
            governance.add(path)
            if path.startswith(ARCHIVE_PREFIX):
                remainder = path[len(ARCHIVE_PREFIX) :]
                if "/" in remainder:
                    archives.add(remainder.split("/", 1)[0])
            elif path.startswith(CHANGES_PREFIX):
                remainder = path[len(CHANGES_PREFIX) :]
                if "/" in remainder:
                    lifecycle = remainder.split("/", 1)[0]
                    active.add(lifecycle)
                    if _hidden_component(remainder.split("/", 1)[1]):
                        hidden.add(path)
            elif path.startswith(EXEMPTIONS_PREFIX):
                exemptions.add(path)
        elif path.startswith(SPEC_PREFIX):
            spec.add(path)
        elif path in PLANNING_NEUTRAL:
            continue
        else:
            production.add(path)
    return Classification(
        active_lifecycles=frozenset(active),
        archives=frozenset(archives),
        exemptions=frozenset(exemptions),
        governance=frozenset(governance),
        spec=frozenset(spec),
        production=frozenset(production),
        hidden=frozenset(hidden),
    )


def check_branch_scope(
    classification: Classification,
    inherited_ids: frozenset[str],
    mode: str,
) -> list[str]:
    if mode not in ("pre-archive", "merge-bound"):
        raise CheckError(f"unknown branch-scope mode: {mode}")
    errors = []
    for path in sorted(classification.hidden):
        errors.append(f"hidden dot-prefixed lifecycle path is not evidence: {path}")
    touched_inherited = classification.active_lifecycles & inherited_ids
    for lifecycle in sorted(touched_inherited):
        errors.append(
            f"branch mutates lifecycle inherited from the comparison base: {lifecycle}"
        )
    new_lifecycles = classification.active_lifecycles - inherited_ids
    if len(new_lifecycles) > 1:
        errors.append(
            "branch adds more than one lifecycle: " + ", ".join(sorted(new_lifecycles))
        )
    if new_lifecycles and classification.exemptions:
        errors.append("branch mixes a lifecycle with a maintenance exemption")
    if len(classification.exemptions) > 1:
        errors.append("branch adds more than one maintenance exemption")
    if len(classification.archives) > 1:
        errors.append("branch adds more than one archived lifecycle")
    governed = classification.spec | classification.production
    if mode == "merge-bound":
        for lifecycle in sorted(new_lifecycles):
            errors.append(
                f"active lifecycle must be archived before merge: {lifecycle}"
            )
        if governed and not (classification.archives or classification.exemptions):
            errors.append(
                "governed paths changed without an archived lifecycle or exemption"
            )
    else:
        if governed and not (new_lifecycles or classification.exemptions):
            errors.append(
                "governed paths changed without a lifecycle or maintenance exemption"
            )
    return errors


def _valid_stamp(year: str, month: str, day: str) -> bool:
    try:
        datetime.date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def check_exemption_manifest(
    name: str, text: str, changed: frozenset[str]
) -> list[str]:
    errors = []
    match = EXEMPTION_NAME.match(name)
    if match is None or not _valid_stamp(*match.groups()[:3]):
        errors.append(
            f"exemption filename must be YYYY-MM-DD-<kebab-case-id>.toml "
            f"with a valid date: {name}"
        )
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        errors.append(f"exemption manifest is not valid TOML: {error}")
        return errors
    if set(manifest) != {"kind", "reason", "paths"}:
        errors.append(
            "exemption manifest must contain exactly kind, reason, and paths"
        )
        return errors
    if manifest["kind"] != "maintenance":
        errors.append("exemption kind must be exactly 'maintenance'")
    reason = manifest["reason"]
    if not isinstance(reason, str) or not reason.strip():
        errors.append("exemption reason must be a non-empty string")
    paths = manifest["paths"]
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        errors.append("exemption paths must be an array of strings")
        return errors
    declared = set(paths)
    for path in sorted(declared):
        if path.startswith(SPEC_PREFIX):
            errors.append(f"exemption can never cover a normative spec path: {path}")
        if path.startswith(GOVERNANCE_PREFIX):
            errors.append(f"exemption can never cover a governance path: {path}")
    if declared != set(changed):
        errors.append(
            "exemption paths must equal the complete changed non-governance set: "
            f"declared {sorted(declared)}, changed {sorted(changed)}"
        )
    return errors


def check_planning_order(commits, lifecycle_id: str) -> list[str]:
    marker = f"{CHANGES_PREFIX}{lifecycle_id}/{MARKER_NAME}"
    proposal = f"{CHANGES_PREFIX}{lifecycle_id}/proposal.md"
    delta = re.compile(
        rf"^{re.escape(CHANGES_PREFIX + lifecycle_id)}/specs/[^/]+/spec\.md$"
    )
    planning_index = None
    for index, (_sha, paths) in enumerate(commits):
        if marker in paths:
            planning_index = index
            break
    if planning_index is None:
        errors = [
            f"branch has no planning commit adding the lifecycle marker {marker}"
        ]
        return errors
    errors = []
    sha, paths = commits[planning_index]
    planning_class = classify_paths(paths)
    if planning_class.production or planning_class.spec:
        errors.append(
            f"planning commit {sha[:12]} must be planning-only but also changes "
            "production or spec paths"
        )
    if proposal not in paths:
        errors.append(f"planning commit {sha[:12]} does not add {proposal}")
    if not any(delta.match(path) for path in paths):
        errors.append(
            f"planning commit {sha[:12]} does not add a requirement delta spec"
        )
    for sha, paths in commits[:planning_index]:
        earlier = classify_paths(paths)
        if earlier.production or earlier.spec:
            errors.append(
                f"commit {sha[:12]} changes production or spec paths before the "
                "planning commit"
            )
    return errors


def check_delta_shape(text: str) -> list[str]:
    errors = []
    for match in re.finditer(r"^## (\S+) Requirements *$", text, re.MULTILINE):
        if match.group(1) not in DELTA_OPERATIONS:
            errors.append(f"unknown delta operation section: {match.group(0)!r}")
    requirements = re.split(r"^### Requirement: *", text, flags=re.MULTILINE)[1:]
    if not requirements:
        errors.append("delta spec declares no requirement blocks")
    for block in requirements:
        name = block.splitlines()[0].strip() if block.splitlines() else "?"
        scenarios = len(re.findall(r"^#### Scenario: ", block, re.MULTILINE))
        if scenarios < 2:
            errors.append(
                f"requirement '{name}' needs at least two scenarios "
                "(positive and negative parity)"
            )
    return errors


def check_artifact_coherence(
    proposal_capabilities: frozenset[str], delta_capabilities: frozenset[str]
) -> list[str]:
    if proposal_capabilities == delta_capabilities:
        return []
    return [
        "proposal capabilities and delta specs disagree: proposal "
        f"{sorted(proposal_capabilities)}, deltas {sorted(delta_capabilities)}"
    ]


def check_tasks(text: str) -> list[str]:
    errors = []
    boxes = 0
    for line in text.splitlines():
        match = CHECKBOX.match(line.strip()) if line.strip().startswith("- [") else None
        if match is None:
            continue
        boxes += 1
        if match.group(1) != "x":
            errors.append(f"task is not complete: {line.strip()!r}")
        if not match.group(2).strip():
            errors.append(f"task has no description: {line.strip()!r}")
    if boxes == 0:
        errors.append("task list contains no checkbox tasks")
    return errors


def check_archive_name(name: str) -> list[str]:
    match = ARCHIVE_NAME.match(name)
    if match is None or not _valid_stamp(*match.groups()[:3]):
        return [
            f"archive directory must be YYYY-MM-DD-<kebab-case-id> with a valid "
            f"date: {name}"
        ]
    return []


def parse_requirement_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pieces = re.split(r"^### Requirement: *", text, flags=re.MULTILINE)[1:]
    for piece in pieces:
        lines = piece.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        body = "\n".join(lines[1:]).strip("\n")
        blocks[name] = f"### Requirement: {name}\n{body}\n"
    return blocks


def parse_delta(text: str) -> dict[str, dict[str, str]]:
    delta: dict[str, dict[str, str]] = {}
    sections = re.split(r"^## (\S+) Requirements *$", text, flags=re.MULTILINE)
    for index in range(1, len(sections), 2):
        operation = sections[index].lower()
        if sections[index].upper() not in DELTA_OPERATIONS:
            raise CheckError(f"unknown delta operation: {sections[index]}")
        blocks = parse_requirement_blocks(sections[index + 1])
        delta.setdefault(operation, {}).update(blocks)
    return delta


def _replay(base: dict[str, str], delta: dict[str, dict[str, str]]) -> dict[str, str]:
    expected = dict(base)
    for name, block in delta.get("added", {}).items():
        if name in expected:
            raise CheckError(f"added requirement already exists in base: {name}")
        expected[name] = block
    for name, block in delta.get("modified", {}).items():
        if name not in expected:
            raise CheckError(f"modified requirement missing from base: {name}")
        expected[name] = block
    for name in delta.get("removed", {}):
        if name not in expected:
            raise CheckError(f"removed requirement missing from base: {name}")
        del expected[name]
    return expected


def check_synchronization(base_specs, head_specs, deltas) -> list[str]:
    errors = []
    capabilities = set(base_specs) | set(head_specs) | set(deltas)
    for capability in sorted(capabilities):
        base = base_specs.get(capability, {})
        head = head_specs.get(capability, {})
        try:
            expected = _replay(base, deltas.get(capability, {}))
        except CheckError as error:
            errors.append(f"{capability}: {error}")
            continue
        if expected != head:
            drifted = sorted(
                name
                for name in set(expected) | set(head)
                if expected.get(name) != head.get(name)
            )
            errors.append(
                f"baseline specs for '{capability}' do not equal the replayed "
                f"archived delta (drift: {', '.join(drifted)})"
            )
    return errors


def _rejects(check, *args) -> bool:
    try:
        return bool(check(*args))
    except CheckError:
        return True


def self_test() -> list[str]:
    marker = f"{CHANGES_PREFIX}example-change/{MARKER_NAME}"
    proposal = f"{CHANGES_PREFIX}example-change/proposal.md"
    delta_path = f"{CHANGES_PREFIX}example-change/specs/cap/spec.md"
    block = "### Requirement: R\nBody.\n"
    outcomes = {
        "malformed-spec": _rejects(check_delta_shape, "## ADDED Requirements\n"),
        "missing-negative-scenario": _rejects(
            check_delta_shape,
            "## ADDED Requirements\n\n### Requirement: R\nBody.\n\n"
            "#### Scenario: only one\n- **WHEN** x\n- **THEN** y\n",
        ),
        "artifact-drift": _rejects(
            check_artifact_coherence, frozenset({"cap-a"}), frozenset({"cap-b"})
        ),
        "planning-order-collapse": _rejects(
            check_planning_order,
            [("a" * 40, frozenset({marker, proposal, delta_path, "crates/x.rs"}))],
            "example-change",
        ),
        "citation-mismatch": _rejects(
            check_citation, ["other-change"], "example-change"
        ),
        "branch-scope-ambiguity": _rejects(
            check_branch_scope,
            classify_paths(
                [marker, f"{CHANGES_PREFIX}second-change/{MARKER_NAME}"]
            ),
            frozenset(),
            "pre-archive",
        ),
        "invalid-exemption": _rejects(
            check_exemption_manifest,
            "2026-07-24-touch-spec.toml",
            'kind = "maintenance"\nreason = "edit"\npaths = ["spec/x.md"]\n',
            frozenset({"spec/x.md"}),
        ),
        "unchecked-task": _rejects(check_tasks, "- [ ] 1.1 Do the thing.\n"),
        "active-merge-state": _rejects(
            check_branch_scope,
            classify_paths([marker, proposal]),
            frozenset(),
            "merge-bound",
        ),
        "malformed-archive": _rejects(
            check_archive_name, "2026-13-40-example-change"
        ),
        "unsynchronized-delta": _rejects(
            check_synchronization,
            {"cap": {}},
            {"cap": {"R": block + "drift\n"}},
            {"cap": {"added": {"R": block}}},
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "real.json"
        target.write_text("{}", encoding="utf-8")
        link = Path(tmp) / "event.json"
        os.symlink(target, link)
        outcomes["symlink-rejection"] = _rejects(load_event, link)
    failures = []
    for control in SELF_TEST_CONTROLS:
        if not outcomes.get(control, False):
            failures.append(f"self-test control failed to reject: {control}")
    return failures


def _git_lines(git, arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        [str(git), *arguments], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise CheckError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _branch_commits(git, merge_base: str):
    output = subprocess.run(
        [
            str(git),
            "log",
            "--reverse",
            "--format=@%H",
            "--name-only",
            f"{merge_base}..HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if output.returncode != 0:
        raise CheckError(f"git log failed: {output.stderr.strip()}")
    commits = []
    sha = None
    paths: set[str] = set()
    for line in output.stdout.splitlines():
        if line.startswith("@"):
            if sha is not None:
                commits.append((sha, frozenset(paths)))
            sha = line[1:].strip()
            paths = set()
        elif line.strip():
            paths.add(line.strip())
    if sha is not None:
        commits.append((sha, frozenset(paths)))
    return commits


def _specs_at(git, revision: str | None, root: Path) -> dict[str, dict[str, str]]:
    specs: dict[str, dict[str, str]] = {}
    if revision is None:
        base = root / "openspec" / "specs"
        for spec_file in sorted(base.glob("*/spec.md")):
            specs[spec_file.parent.name] = parse_requirement_blocks(
                spec_file.read_text(encoding="utf-8")
            )
        return specs
    try:
        listed = _git_lines(
            git, ["ls-tree", "-r", "--name-only", revision, "--", "openspec/specs"]
        )
    except CheckError:
        return specs
    for path in listed:
        match = re.fullmatch(r"openspec/specs/([^/]+)/spec\.md", path)
        if match is None:
            continue
        completed = subprocess.run(
            [str(git), "show", f"{revision}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise CheckError(f"cannot read {path} at {revision}")
        specs[match.group(1)] = parse_requirement_blocks(completed.stdout)
    return specs


def _inherited_lifecycles(git, merge_base: str) -> frozenset[str]:
    try:
        listed = _git_lines(
            git,
            ["ls-tree", "--name-only", merge_base, "--", "openspec/changes/"],
        )
    except CheckError:
        return frozenset()
    inherited = set()
    for entry in listed:
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        if name != "archive":
            inherited.add(name)
    return frozenset(inherited)


def _proposal_capabilities(text: str) -> frozenset[str]:
    capabilities = set()
    section = re.split(r"^### New Capabilities *$", text, flags=re.MULTILINE)
    if len(section) > 1:
        tail = re.split(r"^#{2,3} ", section[1], flags=re.MULTILINE)[0]
        for match in re.finditer(r"^- `([a-z0-9-]+)`", tail, re.MULTILINE):
            capabilities.add(match.group(1))
    return frozenset(capabilities)


def _run_repository_checks(mode: str, base: str, environ) -> list[str]:
    git = resolve_git(dict(environ))
    openspec = resolve_openspec(dict(environ))
    errors = verify_openspec(openspec)
    root = Path.cwd()
    merge_base = resolve_merge_base(git, base)
    changed = frozenset(
        _git_lines(git, ["diff", "--name-only", f"{merge_base}..HEAD"])
    )
    classification = classify_paths(changed)
    inherited = _inherited_lifecycles(git, merge_base)
    errors += check_branch_scope(classification, inherited, mode)
    new_lifecycles = classification.active_lifecycles - inherited

    for lifecycle in sorted(new_lifecycles):
        errors += check_planning_order(_branch_commits(git, merge_base), lifecycle)
        change_dir = root / "openspec" / "changes" / lifecycle
        delta_files = sorted(change_dir.glob("specs/*/spec.md"))
        for delta_file in delta_files:
            errors += check_delta_shape(delta_file.read_text(encoding="utf-8"))
        proposal_file = change_dir / "proposal.md"
        if proposal_file.is_file():
            declared = _proposal_capabilities(
                proposal_file.read_text(encoding="utf-8")
            )
            present = frozenset(f.parent.name for f in delta_files)
            if declared:
                errors += check_artifact_coherence(declared, present)
        errors += _check_citation_from_event(environ, lifecycle)

    for archive in sorted(classification.archives):
        errors += check_archive_name(archive)
        match = ARCHIVE_NAME.match(archive)
        if match is not None:
            # Ordering must hold at merge time too: by then the lifecycle
            # is archived (no longer active), but the branch commits still
            # show whether planning preceded production work.
            errors += check_planning_order(
                _branch_commits(git, merge_base), match.group(4)
            )
        archive_dir = root / "openspec" / "changes" / "archive" / archive
        tasks_file = archive_dir / "tasks.md"
        if tasks_file.is_file():
            errors += check_tasks(tasks_file.read_text(encoding="utf-8"))
        else:
            errors.append(f"archived lifecycle has no tasks.md: {archive}")
        deltas: dict[str, dict[str, dict[str, str]]] = {}
        for delta_file in sorted(archive_dir.glob("specs/*/spec.md")):
            try:
                deltas[delta_file.parent.name] = parse_delta(
                    delta_file.read_text(encoding="utf-8")
                )
            except CheckError as error:
                errors.append(f"{delta_file}: {error}")
        base_specs = _specs_at(git, merge_base, root)
        head_specs = _specs_at(git, None, root)
        errors += check_synchronization(base_specs, head_specs, deltas)
        if match is not None:
            errors += _check_citation_from_event(environ, match.group(4))

    for exemption in sorted(classification.exemptions):
        name = exemption.rsplit("/", 1)[-1]
        exemption_file = root / exemption
        if exemption_file.is_symlink() or not exemption_file.is_file():
            errors.append(
                f"exemption must be a regular non-symlink file: {exemption}"
            )
            continue
        errors += check_exemption_manifest(
            name,
            exemption_file.read_text(encoding="utf-8"),
            classification.spec | classification.production,
        )

    completed = subprocess.run(
        [str(openspec), "validate", "--all", "--strict", "--no-interactive"],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip().splitlines()
        errors.append(
            "strict OpenSpec validation failed: "
            + (detail[-1] if detail else "no diagnostic output")
        )
    return errors


def _check_citation_from_event(environ, lifecycle_id: str) -> list[str]:
    if environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return []
    event_path = environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        return ["pull_request event without GITHUB_EVENT_PATH"]
    payload = load_event(Path(event_path))
    pull_request = payload.get("pull_request")
    body = None
    if isinstance(pull_request, dict):
        body = pull_request.get("body")
    if body is not None and not isinstance(body, str):
        raise CheckError("pull request body must be a string when present")
    return check_citation(extract_citations(body), lifecycle_id)


def main(argv=None, environ=None) -> int:
    environ = os.environ if environ is None else environ
    ns = parse_arguments(sys.argv[1:] if argv is None else argv)
    errors: list[str] = []
    try:
        if ns.self_test:
            errors += self_test()
        if ns.merge_bound:
            errors += _run_repository_checks("merge-bound", ns.base, environ)
        elif ns.pre_archive:
            errors += _run_repository_checks("pre-archive", ns.base, environ)
    except CheckError as error:
        errors.append(str(error))
    for error in errors:
        print(f"check_openspec: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
