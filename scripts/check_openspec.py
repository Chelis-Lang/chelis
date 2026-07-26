"""OpenSpec governance checker (Chelis-owned; stdlib-only, Python 3.11+).

Owning change: `openspec/changes/archive/2026-07-24-adopt-openspec-governance`
(baseline requirements now live in `openspec/specs/`). The shared
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

On GitHub `push` events (the post-merge audit lane) the commit-ordering and
commit-attribution controls are skipped: a squash merge collapses branch
history into one mainline commit, so ordering evidence is owned by the
pull-request lane and local runs. Every diff-shaped control still runs.

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
BASELINE_SPEC = re.compile(r"^openspec/specs/[^/]+/spec\.md$")
GOVERNANCE_ROOT_FILES = frozenset({"openspec/config.yaml"})
SPEC_PREFIX = "spec/"
MARKER_NAME = ".openspec.yaml"
PLANNING_NEUTRAL = frozenset({".gitignore", ".gitattributes"})
RENAMED_GUIDANCE = (
    "RENAMED sections are unsupported; encode renames as REMOVED plus ADDED"
)
# Delta operation section headings. check_delta_shape and parse_delta share
# these so a near-miss heading can never be visible to one path and invisible
# to the other (the divergence that let "## REMOVED  Requirements" fold its
# blocks into an adjacent section). CANONICAL matches only the exact form;
# RESEMBLES matches any level-2 heading ending in "requirements", so a
# wrong-case, extra-space, or RENAMED heading is rejected instead of ignored.
CANONICAL_OPERATION_HEADING = re.compile(
    r"^## (ADDED|MODIFIED|REMOVED) Requirements[ \t]*$", re.MULTILINE
)
_CANONICAL_LINE = re.compile(r"## (?:ADDED|MODIFIED|REMOVED) Requirements[ \t]*")
RESEMBLES_OPERATION_HEADING = re.compile(
    r"^##[ \t]+\S.*requirements[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_RENAMED_LINE = re.compile(
    r"##[ \t]+RENAMED[ \t]+Requirements[ \t]*", re.IGNORECASE
)
REQUIREMENT_HEADING = re.compile(r"^### Requirement:", re.MULTILINE)
# Fenced code block delimiter (CommonMark: up to three leading spaces, then a
# run of >= 3 backticks or tildes). Structural markdown scans mask fenced
# content so an operation heading, requirement, or scenario written inside a
# worked example is never mistaken for live delta structure.
FENCE_DELIMITER = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
ARCHIVE_REQUIRED_FILES = (MARKER_NAME, "proposal.md", "tasks.md")
LIFECYCLE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ARCHIVE_NAME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)$")
EXEMPTION_NAME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)\.toml$"
)
# A checkbox marker is any bracketed text not followed by "(", so a
# markdown link item like "- [Evidence](url)" is not a task while a
# nonstandard marker like "- [wip]" is still policed. The list marker
# covers bullets and ordered items with any interior whitespace because
# GFM renders "-  [ ]" (two spaces) and "1. [ ]" as real checkboxes.
CHECKBOX = re.compile(r"^(?:[-*+]|\d{1,9}[.)])\s+\[([^\]]*)\](?!\() *(.*)$")

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
    "nonstandard-task-marker",
    "active-merge-state",
    "malformed-archive",
    "unsynchronized-delta",
    "unarchived-baseline-mutation",
    "merge-hidden-change",
    "unrecognized-governance-path",
    "unguarded-config-mutation",
    "inherited-exemption-mutation",
    "hidden-archive-path",
    "inherited-archive-mutation",
    "multichar-task-marker",
    "renamed-delta-operation",
    "incomplete-archive",
    "archived-exemption-mix",
    "spaced-task-checkbox",
    "ordered-task-checkbox",
    "malformed-lifecycle-id",
    "lowercase-delta-operation",
    "near-miss-delta-heading",
    "preamble-delta-block",
    "duplicate-delta-requirement",
    "uncovered-baseline-mutation",
    "fenced-scenario-bypass",
    "duplicate-baseline-requirement",
    "nested-hidden-marker",
    "active-archive-mix",
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
    baseline_specs: frozenset[str]
    config: frozenset[str]
    spec: frozenset[str]
    production: frozenset[str]
    hidden: frozenset[str]
    unrecognized: frozenset[str]


def _hidden_component(path: str) -> bool:
    # The lifecycle marker is the only permitted dotfile, and only as the
    # sole path component (a marker at any deeper level, e.g.
    # specs/.openspec.yaml, is a hidden path, not evidence).
    parts = path.split("/")
    for part in parts:
        if not part.startswith("."):
            continue
        if len(parts) == 1 and part == MARKER_NAME:
            continue
        return True
    return False


def classify_paths(paths) -> Classification:
    active: set[str] = set()
    archives: set[str] = set()
    exemptions: set[str] = set()
    governance: set[str] = set()
    baseline: set[str] = set()
    config: set[str] = set()
    spec: set[str] = set()
    production: set[str] = set()
    hidden: set[str] = set()
    unrecognized: set[str] = set()
    for path in paths:
        if path.startswith(GOVERNANCE_PREFIX):
            governance.add(path)
            if path.startswith(ARCHIVE_PREFIX):
                remainder = path[len(ARCHIVE_PREFIX) :]
                if "/" in remainder:
                    archive, rest = remainder.split("/", 1)
                    archives.add(archive)
                    if _hidden_component(rest):
                        hidden.add(path)
                else:
                    unrecognized.add(path)
            elif path.startswith(CHANGES_PREFIX):
                remainder = path[len(CHANGES_PREFIX) :]
                if "/" in remainder:
                    lifecycle = remainder.split("/", 1)[0]
                    active.add(lifecycle)
                    if _hidden_component(remainder.split("/", 1)[1]):
                        hidden.add(path)
                else:
                    unrecognized.add(path)
            elif path.startswith(EXEMPTIONS_PREFIX):
                if "/" in path[len(EXEMPTIONS_PREFIX) :]:
                    unrecognized.add(path)
                else:
                    exemptions.add(path)
            elif BASELINE_SPEC.match(path):
                baseline.add(path)
            elif path in GOVERNANCE_ROOT_FILES:
                config.add(path)
            else:
                unrecognized.add(path)
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
        baseline_specs=frozenset(baseline),
        config=frozenset(config),
        spec=frozenset(spec),
        production=frozenset(production),
        hidden=frozenset(hidden),
        unrecognized=frozenset(unrecognized),
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
        errors.append(f"hidden dot-prefixed governance path is not evidence: {path}")
    for path in sorted(classification.unrecognized):
        errors.append(f"unrecognized governance path is not valid evidence: {path}")
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
    if new_lifecycles and classification.archives:
        # One lifecycle per branch: an active lifecycle alongside an archived
        # one (of any change) is two records. A legitimate create-then-archive
        # nets to archive-only in the endpoint diff, so this never fires on
        # normal archival.
        errors.append("branch mixes an active lifecycle with an archived lifecycle")
    if classification.archives and classification.exemptions:
        errors.append(
            "branch mixes an archived lifecycle with a maintenance exemption"
        )
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
        if classification.baseline_specs and not classification.archives:
            errors.append(
                "baseline openspec/specs changed without an archived lifecycle "
                "in the same change set"
            )
    else:
        if governed and not (
            new_lifecycles or classification.exemptions or classification.archives
        ):
            errors.append(
                "governed paths changed without a lifecycle or maintenance exemption"
            )
        if classification.baseline_specs:
            errors.append(
                "baseline openspec/specs must remain unchanged during "
                "pre-archive validation"
            )
    if classification.config:
        if mode == "merge-bound":
            governing = bool(classification.archives)
        else:
            governing = bool(new_lifecycles or classification.archives)
        if not governing:
            errors.append(
                "openspec configuration changed without a governing lifecycle: "
                + ", ".join(sorted(classification.config))
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


def check_exemption_novelty(exemptions, base_paths) -> list[str]:
    inherited = sorted(frozenset(exemptions) & frozenset(base_paths))
    return [
        "exemption manifest inherited from the comparison base must not "
        f"change: {path}"
        for path in inherited
    ]


def check_archive_novelty(archives, base_archives) -> list[str]:
    inherited = sorted(frozenset(archives) & frozenset(base_archives))
    return [
        "archived lifecycle inherited from the comparison base must not "
        f"change: {name}"
        for name in inherited
    ]


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


def check_commit_coverage(changed, commits) -> list[str]:
    attributed: set[str] = set()
    for _sha, paths in commits:
        attributed.update(paths)
    unattributed = sorted(set(changed) - attributed)
    if unattributed:
        return [
            "changed paths are not attributable to any branch commit "
            "(merge-introduced content bypasses ordering review): "
            + ", ".join(unattributed)
        ]
    return []


def _mask_fences(text: str) -> str:
    """Return text with every fenced-code-block line blanked (characters ->
    spaces, newlines kept) so structural markdown scans never see a heading,
    requirement, or scenario marker that lives inside a code fence. Offsets
    are preserved exactly, so a caller can locate structure in the masked
    view and slice the original text at the same positions."""
    masked: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.split("\n"):
        match = FENCE_DELIMITER.match(line)
        if fence is None:
            if match is not None:
                fence = (match.group(1)[0], len(match.group(1)))
                masked.append(" " * len(line))
            else:
                masked.append(line)
        else:
            char, length = fence
            masked.append(" " * len(line))
            if (
                match is not None
                and match.group(1)[0] == char
                and len(match.group(1)) >= length
                and match.group(2).strip() == ""
            ):
                fence = None
    return "\n".join(masked)


def _delta_sections(text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Split a delta spec into (operation, body, masked_body) sections and
    report every structural problem. Scans run over the fence-masked view so
    a heading written inside a worked example is neither counted as live
    structure nor rejected as a near-miss. A near-miss heading (wrong case,
    extra spaces, RENAMED) and a requirement block preceding the first
    operation section are rejected here so check_delta_shape and parse_delta
    cannot disagree about what the schema is."""
    errors: list[str] = []
    masked = _mask_fences(text)
    for match in RESEMBLES_OPERATION_HEADING.finditer(masked):
        heading = match.group(0)
        if _CANONICAL_LINE.fullmatch(heading):
            continue
        if _RENAMED_LINE.fullmatch(heading.strip()):
            errors.append(RENAMED_GUIDANCE)
        else:
            errors.append(
                f"malformed delta operation section: {heading.strip()!r}"
            )
    canonical = list(CANONICAL_OPERATION_HEADING.finditer(masked))
    if canonical and REQUIREMENT_HEADING.search(masked[: canonical[0].start()]):
        errors.append(
            "requirement block precedes the first delta operation section "
            "and would not be replayed"
        )
    bounds = [match.start() for match in canonical] + [len(text)]
    sections = [
        (
            match.group(1),
            text[match.end() : bounds[index + 1]],
            masked[match.end() : bounds[index + 1]],
        )
        for index, match in enumerate(canonical)
    ]
    return sections, errors


def check_delta_shape(text: str) -> list[str]:
    sections, errors = _delta_sections(text)
    seen: set[str] = set()
    blocks = 0
    for operation, _body, masked_body in sections:
        # Split and count over the fence-masked body: a '#### Scenario:' line
        # hidden inside a code fence must not satisfy the parity rule.
        for block in re.split(
            r"^### Requirement: *", masked_body, flags=re.MULTILINE
        )[1:]:
            blocks += 1
            lines = block.splitlines()
            name = lines[0].strip() if lines else "?"
            if name in seen:
                errors.append(f"duplicate requirement block: {name!r}")
            seen.add(name)
            # A REMOVED section names the requirement and its reason; it has
            # no changed behavior to demonstrate, so it is exempt from the
            # positive/negative scenario parity rule.
            if operation == "REMOVED":
                continue
            scenarios = len(re.findall(r"^#### Scenario: ", block, re.MULTILINE))
            if scenarios < 2:
                errors.append(
                    f"requirement '{name}' needs at least two scenarios "
                    "(positive and negative parity)"
                )
    if blocks == 0:
        errors.append("delta spec declares no requirement blocks")
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
        stripped = line.strip()
        match = CHECKBOX.match(stripped)
        if match is None:
            continue
        boxes += 1
        marker = match.group(1)
        if not marker.strip():
            errors.append(f"task is not complete: {stripped!r}")
        elif marker != "x":
            errors.append(
                f"task marker must be exactly '[x]' when complete: {stripped!r}"
            )
        if not match.group(2).strip():
            errors.append(f"task has no description: {stripped!r}")
    if boxes == 0:
        errors.append("task list contains no checkbox tasks")
    return errors


def check_archive_deltas(
    archive_dir: Path, archive: str
) -> tuple[list[str], dict[str, dict[str, dict[str, str]]]]:
    errors: list[str] = []
    deltas: dict[str, dict[str, dict[str, str]]] = {}
    delta_files = sorted(archive_dir.glob("specs/*/spec.md"))
    if not delta_files:
        errors.append(
            f"archived lifecycle has no requirement delta specs: {archive}"
        )
    for delta_file in delta_files:
        text = delta_file.read_text(encoding="utf-8")
        errors += check_delta_shape(text)
        try:
            deltas[delta_file.parent.name] = parse_delta(text)
        except CheckError as error:
            errors.append(f"{delta_file}: {error}")
    return errors, deltas


def check_archive_completeness(archive_dir: Path, archive: str) -> list[str]:
    return [
        f"archived lifecycle is missing {name}: {archive}"
        for name in ARCHIVE_REQUIRED_FILES
        if not (archive_dir / name).is_file()
    ]


def check_lifecycle_name(name: str) -> list[str]:
    if LIFECYCLE_NAME.match(name) is None:
        return [f"lifecycle identifier must be lowercase kebab-case: {name}"]
    return []


def check_archive_name(name: str) -> list[str]:
    match = ARCHIVE_NAME.match(name)
    if match is None or not _valid_stamp(*match.groups()[:3]):
        return [
            f"archive directory must be YYYY-MM-DD-<kebab-case-id> with a valid "
            f"date: {name}"
        ]
    return []


def parse_requirement_blocks(text: str) -> dict[str, str]:
    # Locate requirement headings in the fence-masked view so a
    # '### Requirement:' inside a worked example is body text, not a real
    # block, then slice the original text so the block body keeps its literal
    # (fenced) content for drift comparison. Duplicate names are rejected
    # rather than silently collapsed last-wins.
    masked = _mask_fences(text)
    headings = list(re.finditer(r"^### Requirement: *", masked, flags=re.MULTILINE))
    blocks: dict[str, str] = {}
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        lines = text[start:end].splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        body = "\n".join(lines[1:]).strip("\n")
        if name in blocks:
            raise CheckError(f"duplicate requirement block: {name}")
        blocks[name] = f"### Requirement: {name}\n{body}\n"
    return blocks


def parse_delta(text: str) -> dict[str, dict[str, str]]:
    sections, errors = _delta_sections(text)
    if errors:
        raise CheckError(errors[0])
    delta: dict[str, dict[str, str]] = {}
    for operation, body, _masked_body in sections:
        target = delta.setdefault(operation.lower(), {})
        for name, block in parse_requirement_blocks(body).items():
            if name in target:
                raise CheckError(f"duplicate requirement block: {name}")
            target[name] = block
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


def check_baseline_delta_coverage(
    baseline_specs: frozenset[str], delta_capabilities: frozenset[str]
) -> list[str]:
    # Requirement-block replay leaves a baseline file's non-requirement
    # prose (title, Purpose) unchecked, so an archived change to one
    # capability could smuggle edits into an unrelated capability's
    # baseline. Tie every changed baseline file to a delta: a capability
    # whose baseline changed must own a delta in the same change set.
    errors = []
    for path in sorted(baseline_specs):
        match = BASELINE_SPEC.match(path)
        capability = path[len("openspec/specs/") :].split("/", 1)[0]
        if match is None or capability not in delta_capabilities:
            errors.append(
                "baseline spec changed without a corresponding archived delta: "
                f"{path}"
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
        "nonstandard-task-marker": _rejects(
            check_tasks, "- [x] 1.1 Done.\n- [~] 1.2 Deferred.\n"
        ),
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
        "unarchived-baseline-mutation": _rejects(
            check_branch_scope,
            classify_paths(["openspec/specs/cap/spec.md"]),
            frozenset(),
            "merge-bound",
        ),
        "merge-hidden-change": _rejects(
            check_commit_coverage,
            frozenset({"crates/x.rs"}),
            [("a" * 40, frozenset({marker}))],
        ),
        "unrecognized-governance-path": _rejects(
            check_branch_scope,
            classify_paths(["openspec/notes/hack.txt"]),
            frozenset(),
            "pre-archive",
        ),
        "unguarded-config-mutation": _rejects(
            check_branch_scope,
            classify_paths(["openspec/config.yaml"]),
            frozenset(),
            "merge-bound",
        ),
        "inherited-exemption-mutation": _rejects(
            check_exemption_novelty,
            frozenset({"openspec/exemptions/2026-01-01-old.toml"}),
            frozenset({"openspec/exemptions/2026-01-01-old.toml"}),
        ),
        "hidden-archive-path": _rejects(
            check_branch_scope,
            classify_paths(
                ["openspec/changes/archive/2026-07-24-x-change/.hidden/evil.md"]
            ),
            frozenset(),
            "merge-bound",
        ),
        "inherited-archive-mutation": _rejects(
            check_archive_novelty,
            frozenset({"2026-01-01-old-change"}),
            frozenset({"2026-01-01-old-change"}),
        ),
        "multichar-task-marker": _rejects(
            check_tasks, "- [x] 1.1 Done.\n- [wip] 1.2 In progress.\n"
        ),
        "renamed-delta-operation": _rejects(
            check_delta_shape,
            "## RENAMED Requirements\n\n### Requirement: R\nBody.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n",
        ),
        "incomplete-archive": _rejects(
            check_archive_completeness,
            Path("nonexistent-archive-fixture"),
            "2026-07-24-x-change",
        ),
        "archived-exemption-mix": _rejects(
            check_branch_scope,
            classify_paths(
                [
                    f"{ARCHIVE_PREFIX}2026-07-24-x-change/proposal.md",
                    "openspec/exemptions/2026-07-24-side-fix.toml",
                ]
            ),
            frozenset(),
            "merge-bound",
        ),
        "spaced-task-checkbox": _rejects(
            check_tasks, "- [x] 1.1 Done.\n-  [ ] 1.2 Hidden.\n"
        ),
        "ordered-task-checkbox": _rejects(
            check_tasks, "- [x] 1.1 Done.\n1. [ ] 1.2 Hidden.\n"
        ),
        "malformed-lifecycle-id": _rejects(check_lifecycle_name, "Evil_Change"),
        "lowercase-delta-operation": _rejects(
            parse_delta, "## added Requirements\n\n### Requirement: R\nBody.\n"
        ),
        "near-miss-delta-heading": _rejects(
            check_delta_shape,
            "## ADDED Requirements\n\n### Requirement: R\nBody.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n\n"
            "## REMOVED  Requirements\n\n### Requirement: S\nGone.\n",
        ),
        "preamble-delta-block": _rejects(
            check_delta_shape,
            "### Requirement: Ghost\nBody.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n\n"
            "## ADDED Requirements\n\n### Requirement: R\nBody.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n",
        ),
        "duplicate-delta-requirement": _rejects(
            check_delta_shape,
            "## ADDED Requirements\n\n### Requirement: R\nFirst.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n\n"
            "### Requirement: R\nSecond.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n",
        ),
        "uncovered-baseline-mutation": _rejects(
            check_baseline_delta_coverage,
            frozenset({"openspec/specs/other-cap/spec.md"}),
            frozenset({"cap"}),
        ),
        "fenced-scenario-bypass": _rejects(
            check_delta_shape,
            "## ADDED Requirements\n\n### Requirement: R\nBody.\n\n"
            "```\n#### Scenario: fenced positive\n"
            "#### Scenario: fenced negative\n```\n",
        ),
        "duplicate-baseline-requirement": _rejects(
            parse_requirement_blocks,
            "### Requirement: R\nFirst.\n\n### Requirement: R\nSecond.\n",
        ),
        "nested-hidden-marker": _rejects(
            check_branch_scope,
            classify_paths(
                [f"{ARCHIVE_PREFIX}2026-07-24-x-change/specs/{MARKER_NAME}"]
            ),
            frozenset(),
            "merge-bound",
        ),
        "active-archive-mix": _rejects(
            check_branch_scope,
            classify_paths(
                [marker, f"{ARCHIVE_PREFIX}2026-07-24-other-change/proposal.md"]
            ),
            frozenset(),
            "pre-archive",
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
            # NUL sentinel: no committed path can begin with %x00, so a
            # crafted filename cannot forge a commit boundary.
            "--format=%x00%H",
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
        if line.startswith("\0"):
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
            try:
                specs[spec_file.parent.name] = parse_requirement_blocks(
                    spec_file.read_text(encoding="utf-8")
                )
            except CheckError as error:
                raise CheckError(
                    f"openspec/specs/{spec_file.parent.name}/spec.md: {error}"
                ) from error
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
        try:
            specs[match.group(1)] = parse_requirement_blocks(completed.stdout)
        except CheckError as error:
            raise CheckError(f"{path} at {revision}: {error}") from error
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


def _base_archives(git, merge_base: str) -> frozenset[str]:
    try:
        listed = _git_lines(
            git,
            ["ls-tree", "--name-only", merge_base, "--", ARCHIVE_PREFIX],
        )
    except CheckError:
        return frozenset()
    return frozenset(entry.rstrip("/").rsplit("/", 1)[-1] for entry in listed)


def _base_exemptions(git, merge_base: str) -> frozenset[str]:
    try:
        listed = _git_lines(
            git,
            ["ls-tree", "-r", "--name-only", merge_base, "--", EXEMPTIONS_PREFIX],
        )
    except CheckError:
        return frozenset()
    return frozenset(listed)


def _proposal_capabilities(text: str) -> frozenset[str]:
    capabilities = set()
    for match in re.finditer(
        r"^### (?:New|Modified) Capabilities *$", text, re.MULTILINE
    ):
        tail = re.split(r"^#{2,3} ", text[match.end() :], flags=re.MULTILINE)[0]
        for bullet in re.finditer(r"^- `([a-z0-9-]+)`", tail, re.MULTILINE):
            capabilities.add(bullet.group(1))
    return frozenset(capabilities)


def resolve_repo_root(git) -> Path:
    completed = subprocess.run(
        [str(git), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CheckError(
            "cannot resolve the repository root: "
            "git rev-parse --show-toplevel failed"
        )
    return Path(completed.stdout.strip())


def _run_repository_checks(mode: str, base: str, environ) -> list[str]:
    git = resolve_git(dict(environ))
    openspec = resolve_openspec(dict(environ))
    errors = verify_openspec(openspec)
    if errors:
        # Fail closed before trusting any output from a wrong-version tool.
        return errors
    root = resolve_repo_root(git)
    merge_base = resolve_merge_base(git, base)
    changed = frozenset(
        _git_lines(git, ["diff", "--name-only", f"{merge_base}..HEAD"])
    )
    classification = classify_paths(changed)
    inherited = _inherited_lifecycles(git, merge_base)
    errors += check_branch_scope(classification, inherited, mode)
    errors += check_archive_novelty(
        classification.archives, _base_archives(git, merge_base)
    )
    commits = _branch_commits(git, merge_base)
    # A GitHub push event is the post-merge audit lane: a squash merge
    # collapses branch history into one mainline commit, so ordering and
    # attribution evidence exists only on the pull-request lane and in
    # local runs. Every diff-shaped control below still runs on push.
    ordered_history = environ.get("GITHUB_EVENT_NAME") != "push"
    if ordered_history:
        errors += check_commit_coverage(changed, commits)
    new_lifecycles = classification.active_lifecycles - inherited

    for lifecycle in sorted(new_lifecycles):
        errors += check_lifecycle_name(lifecycle)
        if ordered_history:
            errors += check_planning_order(commits, lifecycle)
        change_dir = root / "openspec" / "changes" / lifecycle
        delta_files = sorted(change_dir.glob("specs/*/spec.md"))
        for delta_file in delta_files:
            errors += check_delta_shape(delta_file.read_text(encoding="utf-8"))
        proposal_file = change_dir / "proposal.md"
        declared = frozenset()
        if proposal_file.is_file():
            declared = _proposal_capabilities(
                proposal_file.read_text(encoding="utf-8")
            )
        present = frozenset(f.parent.name for f in delta_files)
        if declared or present:
            errors += check_artifact_coherence(declared, present)
        errors += _check_citation_from_event(environ, lifecycle)

    deltas: dict[str, dict[str, dict[str, str]]] = {}
    for archive in sorted(classification.archives):
        errors += check_archive_name(archive)
        match = ARCHIVE_NAME.match(archive)
        if match is not None and ordered_history:
            # Ordering must hold at merge time too: by then the lifecycle
            # is archived (no longer active), but the branch commits still
            # show whether planning preceded production work.
            errors += check_planning_order(commits, match.group(4))
        archive_dir = root / "openspec" / "changes" / "archive" / archive
        errors += check_archive_completeness(archive_dir, archive)
        tasks_file = archive_dir / "tasks.md"
        if tasks_file.is_file():
            errors += check_tasks(tasks_file.read_text(encoding="utf-8"))
        archive_errors, archive_deltas = check_archive_deltas(archive_dir, archive)
        errors += archive_errors
        deltas.update(archive_deltas)
        proposal_file = archive_dir / "proposal.md"
        if proposal_file.is_file():
            declared = _proposal_capabilities(
                proposal_file.read_text(encoding="utf-8")
            )
            present = frozenset(
                f.parent.name for f in archive_dir.glob("specs/*/spec.md")
            )
            if declared or present:
                errors += check_artifact_coherence(declared, present)
        if match is not None:
            errors += _check_citation_from_event(environ, match.group(4))
    if classification.archives or classification.baseline_specs:
        # Baseline openspec/specs must always equal the comparison base
        # plus the replayed archived deltas; a diff with baseline edits
        # but no archive replays an empty delta and fails on any drift.
        base_specs = _specs_at(git, merge_base, root)
        head_specs = _specs_at(git, None, root)
        errors += check_synchronization(base_specs, head_specs, deltas)
        if classification.archives:
            errors += check_baseline_delta_coverage(
                classification.baseline_specs, frozenset(deltas)
            )

    base_exemptions = _base_exemptions(git, merge_base)
    errors += check_exemption_novelty(classification.exemptions, base_exemptions)
    for exemption in sorted(classification.exemptions - base_exemptions):
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
