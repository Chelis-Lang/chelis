#!/usr/bin/env python3
"""Decide whether a CI run touches only documentation/prose.

The CI workflow skips the heavy per-PR jobs (Integration, macOS Smoke,
SMT Feature Build, Backend Sanitizers, Hull Conformance Gate) on
docs-only pull requests via a *job-level* `if`, never `paths-ignore`
(chelis#419). A required check that is path-filtered out never reports
its status context, so branch protection waits on it forever and the PR
can never merge; a required check whose *job* is skipped by an `if`
reports its context as success, which satisfies branch protection. So
the decision has to be a job output, computed here, fed into each heavy
job's `if`.

This is the conservative half of that contract: a run is "docs only"
only when the changed-file set is NON-EMPTY and EVERY path matches the
docs allowlist. An empty list (a diff that produced nothing, a merge
with no file delta, a parse failure) is treated as NOT docs-only so the
full suite still runs -- the safe direction, since the cost of a false
"not docs only" is a redundant full run, while a false "docs only"
would let a code change skip the heavy gate.

Usage:
    git diff --name-only <base>..<head> | python3 scripts/ci_detect_docs_only.py

Writes `docs_only=<bool>`, `rejection_authority_changed=<bool>`, and
`diagnostic_kind_changed=<bool>` to
the file named by `$GITHUB_OUTPUT` (the GitHub Actions step-output
mechanism); if that env var is unset it prints both lines to stdout so the
script is runnable and testable off CI. Exit status is always 0. An empty or
unreadable change set fails safe in both directions: it runs the full build
and the network-backed rejection-authority liveness check.
"""

from __future__ import annotations

import os
import sys
from pathlib import PurePosixPath

# Paths that are pure documentation/prose: a change confined to these is
# safe to skip the heavy build/test jobs for. Exact-name files plus
# suffix and directory-prefix rules. Conservative on purpose -- anything
# not matched here counts as code and forces the full suite.
DOC_SUFFIXES: tuple[str, ...] = (".md", ".markdown")
DOC_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "CODEOWNERS",
        "NOTICE",
        ".gitignore",
    }
)
# Directory prefixes whose entire contents are prose. `docs/` holds the
# mdBook source and investigations; an mdBook-only change still has its
# own Docs job (mdbook build + the skill_suite validator), which is one
# of the always-run jobs, so skipping the heavy jobs for a docs/-only
# change keeps coverage. `openspec/changes/` is planning prose, but every
# change carries a non-Markdown `.openspec.yaml`, which alone would force
# the full matrix; the openspec-validate workflow keys on `openspec/**`
# and still runs. `openspec/config.yaml` stays code -- it configures the
# tool, not one change. Trailing slash is required so a sibling file
# like `docsignore` does not match.
DOC_DIR_PREFIXES: tuple[str, ...] = ("docs/", "openspec/changes/")

# Inputs whose edits could authorize a new [05-UNS-5] issue citation or weaken
# its checker. The CI workflow itself is included so a would-be bypass to the
# job is exercised by the job in the same pull request.
REJECTION_AUTHORITY_PATHS: frozenset[str] = frozenset(
    {
        ".github/workflows/ci.yml",
        "crates/chelis-types/src/lib.rs",
        "crates/chelis-types/src/rejection_registry_generated.rs",
        "crates/chelis-types/src/unsupported.rs",
        "scripts/capacity_census_liveness.py",
        "scripts/check_rejection_authority_boundary.py",
        "scripts/ci_detect_docs_only.py",
        "scripts/generate_rejection_registries.py",
        "scripts/test_check_rejection_authority_boundary.py",
        "scripts/validate_rejection_issue_manifest.py",
        "spec/design/loud_unsupported_issue_manifest.json",
    }
)

# Every source or control whose edit could reopen same-crate Diagnostic.kind
# construction/mutation or make the closed-vocabulary mutation oracle stop
# exercising its real owner. The required diagnostic-kind-oracle job consumes
# this output. Including the detector and workflow makes bypass edits
# self-triggering.
DIAGNOSTIC_KIND_PATHS: frozenset[str] = frozenset(
    {
        ".github/workflows/ci.yml",
        "crates/chelis-compiler-api/src/context.rs",
        "crates/chelis-compiler-api/src/lib.rs",
        "crates/chelis-compiler-api/src/schema.rs",
        "crates/chelis-compiler-api/tests/diagnostic_kind_pipeline.rs",
        "crates/chelis-vocab/src/lib.rs",
        "scripts/ci_detect_docs_only.py",
        "scripts/diagnostic_kind_oracle.py",
        "scripts/test_ci_detect_docs_only.py",
        "scripts/test_diagnostic_kind_oracle.py",
        "scripts/test_gate.py",
    }
)


def is_doc_path(path: str) -> bool:
    """True if `path` is documentation/prose under the allowlist."""
    norm = path.strip().strip('"')
    if not norm:
        return False
    # Normalize to forward slashes; git emits POSIX separators already,
    # but be explicit so a stray backslash path is treated as code (it
    # will not match a doc rule) rather than crashing.
    posix = PurePosixPath(norm)
    name = posix.name
    if name in DOC_EXACT_NAMES:
        return True
    if any(name.endswith(suffix) for suffix in DOC_SUFFIXES):
        return True
    if any(norm.startswith(prefix) for prefix in DOC_DIR_PREFIXES):
        return True
    return False


def is_docs_only(paths: list[str]) -> bool:
    """A change is docs-only iff the path set is non-empty and every
    path is a doc path. Empty input -> False (run the full suite)."""
    cleaned = [p.strip().strip('"') for p in paths if p.strip()]
    if not cleaned:
        return False
    return all(is_doc_path(p) for p in cleaned)


def rejection_authority_changed(paths: list[str]) -> bool:
    """Whether the diff must run live validation; empty input fails safe."""
    cleaned = [p.strip().strip('"') for p in paths if p.strip()]
    if not cleaned:
        return True
    return any(path in REJECTION_AUTHORITY_PATHS for path in cleaned)


def diagnostic_kind_changed(paths: list[str]) -> bool:
    """Whether the diff must run the C2.2 mutation oracle; empty fails safe."""
    cleaned = [p.strip().strip('"') for p in paths if p.strip()]
    if not cleaned:
        return True
    return any(path in DIAGNOSTIC_KIND_PATHS for path in cleaned)


def _emit(
    docs_only: bool,
    authority_changed: bool,
    diagnostic_changed: bool,
) -> None:
    lines = [
        f"docs_only={'true' if docs_only else 'false'}",
        "rejection_authority_changed="
        f"{'true' if authority_changed else 'false'}",
        "diagnostic_kind_changed="
        f"{'true' if diagnostic_changed else 'false'}",
    ]
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    # Always echo to stdout too so the decision is visible in the CI log
    # and the script is testable off CI.
    print("\n".join(lines))


def main(argv: list[str]) -> int:
    paths = sys.stdin.read().splitlines()
    _emit(
        is_docs_only(paths),
        rejection_authority_changed(paths),
        diagnostic_kind_changed(paths),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
