#!/usr/bin/env python3
"""Report committed Phase 3 required-test changes alongside existing guards.

The report is a review cue, not a behavioral acceptance oracle. Definition
digests, comparator checks, receipts, and mutation controls remain in force.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from faithful_observation_phase3_oracle import test_definition_spans


ROOT = Path(__file__).resolve().parents[1]
ORACLE = "scripts/faithful_observation_phase3_oracle.py"
DOCTRINE = (
    "Guard artifact: changing a guard merely to accept an edit is not a repair. "
    "Changed definitions require review and independent behavior evidence; "
    "this report does not replace the Phase 3 oracle."
)
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
TEST_DECLARATION = re.compile(
    r"(?P<attrs>(?:#\[[^\]]+\]\s*)+)fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def read_required_tests(source: str) -> dict[str, set[str]]:
    """Read the literal inventory without importing either historical module."""
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"cannot parse Phase 3 inventory: {error}") from error
    bindings: dict[str, list[ast.expr]] = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(value)
    inventories = bindings.get("REQUIRED_TESTS", [])
    if len(inventories) != 1 or not isinstance(inventories[0], ast.Dict):
        raise ValueError("Phase 3 REQUIRED_TESTS must be one literal dictionary")
    inventory = inventories[0]
    if not inventory.keys:
        raise ValueError("Phase 3 required-test inventory must not be empty")
    parents = {child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(module):
        if not isinstance(node, ast.Name) or node.id != "REQUIRED_TESTS":
            continue
        parent = parents[node]
        if isinstance(node.ctx, ast.Store):
            if getattr(parent, "value", None) is not inventory or (
                isinstance(parent, ast.Assign) and len(parent.targets) != 1
            ):
                raise ValueError("REQUIRED_TESTS cannot be rebound or declared through an alias")
        else:
            call = parents.get(parent)
            if not (
                isinstance(parent, ast.Attribute)
                and parent.attr in {"items", "keys", "values", "get"}
                and isinstance(call, ast.Call) and call.func is parent
            ):
                raise ValueError("REQUIRED_TESTS must stay literal; mutation or aliased access is unsupported")
    result: dict[str, set[str]] = {}
    for key, value in zip(inventory.keys, inventory.values):
        if not isinstance(key, ast.Name) or len(bindings.get(key.id, [])) != 1:
            raise ValueError("required-test source must name one literal Path binding")
        path_value = bindings[key.id][0]
        if (
            not isinstance(path_value, ast.Call)
            or not isinstance(path_value.func, ast.Name)
            or path_value.func.id != "Path"
            or len(path_value.args) != 1
            or path_value.keywords
            or not isinstance(path_value.args[0], ast.Constant)
            or not isinstance(path_value.args[0].value, str)
        ):
            raise ValueError(f"required-test source {key.id} must be a literal Path")
        path = path_value.args[0].value
        parsed_path = PurePosixPath(path)
        if (
            parsed_path.is_absolute() or ".." in parsed_path.parts
            or str(parsed_path) != path or parsed_path.suffix != ".rs"
            or "\\" in path or any(ord(c) < 32 for c in path)
        ):
            raise ValueError(f"invalid repository-relative Rust source: {path!r}")
        if path in result or not isinstance(value, ast.Set) or not value.elts:
            raise ValueError(f"required tests for {path} must be one nonempty literal set")
        names = []
        for item in value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str) or not NAME.fullmatch(item.value):
                raise ValueError(f"invalid literal required-test identity in {path}")
            names.append(item.value)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate required-test identity in {path}")
        result[path] = set(names)
    return result


def git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.decode(errors="replace").strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ValueError(f"cannot read committed Phase 3 evidence: {detail}") from error


def commit(repo: Path, ref: str) -> str:
    result = git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", result):
        raise ValueError(f"not an exact commit: {ref!r}")
    return result


def source_at(repo: Path, revision: str, path: str, *, optional: bool = False) -> str:
    if optional and not git(repo, "ls-tree", "--name-only", "-z", revision, "--", path):
        return ""
    return git(repo, "show", f"{revision}:{path}").decode("utf-8")


def definitions(source: str, required: set[str]) -> dict[str, tuple[str, list[int]]]:
    # Reuse the oracle's definition boundaries. Refuse its otherwise ambiguous
    # name-only identity when more than one declaration bears the same name.
    names = Counter(
        match.group("name") for match in TEST_DECLARATION.finditer(source)
        if "#[test]" in match.group("attrs")
    )
    duplicates = sorted(name for name, count in names.items() if name in required and count != 1)
    if duplicates:
        raise ValueError(f"ambiguous Phase 3 test definitions: {duplicates}")
    return {
        name: (
            source[start:end],
            [source.count("\n", 0, start) + 1, source.count("\n", 0, end - 1) + 1],
        )
        for name, (start, end, _, _) in test_definition_spans(source).items()
        if name in required
    }


def changed_tests(repo: Path, base_ref: str, candidate_ref: str = "HEAD") -> dict:
    comparison = commit(repo, base_ref)
    candidate = commit(repo, candidate_ref)
    bases = git(repo, "merge-base", "--all", comparison, candidate).decode().splitlines()
    if len(bases) != 1:
        raise ValueError("Phase 3 comparison requires one unique merge base")
    base = bases[0]
    inventories = {
        revision: read_required_tests(source_at(repo, revision, ORACLE))
        for revision in {base, candidate}
    }
    before_inventory, after_inventory = inventories[base], inventories[candidate]
    paths = sorted(set(before_inventory) | set(after_inventory))
    snapshots = {
        revision: {
            path: definitions(
                source_at(repo, revision, path, optional=True),
                before_inventory.get(path, set()) | after_inventory.get(path, set()),
            )
            for path in paths
        }
        for revision in {base, candidate}
    }
    changes, problems = [], []
    for path in paths:
        before_required = before_inventory.get(path, set())
        after_required = after_inventory.get(path, set())
        for name in sorted(before_required | after_required):
            before = snapshots[base][path].get(name)
            after = snapshots[candidate][path].get(name)
            kinds = []
            if name not in before_required:
                kinds.append("newly required")
            elif name not in after_required:
                kinds.append("no longer required")
            if before is None and after is not None:
                kinds.append("definition added")
            elif before is not None and after is None:
                kinds.append("definition removed")
            elif before is not None and after is not None and before[0] != after[0]:
                kinds.append("definition changed")
            if name in after_required and after is None:
                problems.append(f"candidate is missing required definition: {path}::{name}")
            if kinds:
                changes.append({
                    "path": path, "test": name, "changes": kinds,
                    "before_lines": before[1] if before is not None else None,
                    "after_lines": after[1] if after is not None else None,
                })
    return {
        "version": 1, "comparison_ref": comparison, "base": base, "candidate": candidate,
        "changes": changes, "problems": problems, "doctrine": DOCTRINE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="comparison revision; derive its merge base with candidate")
    parser.add_argument("--candidate", default="HEAD", help="committed candidate revision (default HEAD)")
    parser.add_argument("--output", type=Path, required=True, help="JSON report to publish")
    args = parser.parse_args(argv)
    try:
        args.output.unlink(missing_ok=True)
        result = changed_tests(ROOT, args.base, args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, OSError) as error:
        print(f"PHASE 3 TEST CHANGE REPORT: FAIL: {error}", file=sys.stderr)
        return 1
    print(f'Phase 3 required-test comparison: {result["base"]} -> {result["candidate"]}')
    for row in result["changes"]:
        print(f'  {row["path"]}::{row["test"]}: {", ".join(row["changes"])}')
    print(DOCTRINE)
    for problem in result["problems"]:
        print(problem, file=sys.stderr)
    if result["problems"]:
        print("PHASE 3 TEST CHANGE REPORT: FAIL", file=sys.stderr)
        return 1
    print(f'PHASE 3 TEST CHANGE REPORT: PASS ({len(result["changes"])} changed identities; behavior not certified)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
