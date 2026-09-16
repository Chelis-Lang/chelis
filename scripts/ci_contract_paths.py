#!/usr/bin/env python3
"""Classify paths that control required pull-request validation."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


CI_CONTRACT_EXACT_PATHS: frozenset[str] = frozenset(
    {
        ".config/ci-change-owned-durations.json",
        ".config/ci-test-targets.toml",
        "AGENTS.md",
        "agent-skills/redteam-exec/SKILL.md",
        "docs/ci_validation.md",
        "docs/guard_changes_for_pr_authors.md",
        "spec/design/guard_artifact_proposal_assessment.md",
        "scripts/changelog.py",
        "scripts/check_agent_skills.py",
        "scripts/check_configuration_closure.py",
        "scripts/check_rejection_authority_boundary.py",
        "scripts/diagnostic_kind_oracle.py",
        "scripts/dtype_phase4b_oracle.py",
        "scripts/gate.py",
        "scripts/phase3_test_change_report.py",
        "scripts/phase4b_change_report.py",
        "scripts/test_changelog.py",
        "scripts/test_change_owned_workflow.py",
        "scripts/test_check_agent_skills.py",
        "scripts/test_diagnostic_kind_oracle.py",
        "scripts/test_dtype_phase4b_oracle.py",
        "scripts/test_gate.py",
        "scripts/test_hosted_validation.py",
        "scripts/test_phase3_test_change_report.py",
        "scripts/test_phase4b_change_report.py",
        "scripts/test_pr_workflow_routing.py",
        "scripts/test_regenerate_conformance_assets.py",
        "scripts/test_release_workflow_pyo3_isolation.py",
        "scripts/verify_release_smt.py",
    }
)
CI_CONTRACT_PREFIXES: tuple[str, ...] = (
    ".github/actions/",
    ".github/workflows/",
    "scripts/ci_",
    "scripts/test_ci_",
)


def normalized_paths(paths: Sequence[str]) -> list[str]:
    return [path.strip().strip('"') for path in paths if path.strip()]


def is_ci_contract_path(path: str) -> bool:
    normalized = path.strip().strip('"')
    return bool(normalized) and (
        normalized in CI_CONTRACT_EXACT_PATHS
        or any(
            normalized.startswith(prefix)
            for prefix in CI_CONTRACT_PREFIXES
        )
    )


def ci_contract_changed(paths: Sequence[str]) -> bool:
    """Whether required-check policy changed; empty input fails safe."""
    cleaned = normalized_paths(paths)
    return not cleaned or any(is_ci_contract_path(path) for path in cleaned)


def reuse_eligibility(paths: Sequence[str]) -> tuple[bool, list[str]]:
    blockers = sorted(
        path for path in normalized_paths(paths) if is_ci_contract_path(path)
    )
    return not blockers, blockers


def main() -> int:
    changed = ci_contract_changed(sys.stdin.read().splitlines())
    line = f"ci_contract_changed={'true' if changed else 'false'}"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
