#!/usr/bin/env python3
"""Publish committed Phase 4B changes for enforcing contract acknowledgements.

Required identities, literal anchors and semantic registrations remain separate
requirements of the authoritative Phase 4B oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Callable

from builtin_atom_registry import parse_registry, ROW
from ci_change_owned import resolve_pr_commits
from phase4b_contract_text import ATOM_START, OracleError, frozen_region, normalize_frozen_block, strict_atom_block


ROOT = Path(__file__).resolve().parents[1]
ORACLE = "scripts/dtype_phase4b_oracle.py"
ATOM_FILES = ("spec/04-type-system.md", "spec/05-risc-primitives.md")
BUILTINS = "spec/registry/builtin_semantic_identities.md"
TABLES = ("CONTRACT_FILES", "REQUIRED_ATOMS", "REQUIRED_REGIONS", "OP_MANIFEST_REGISTRY_FILES")
HISTORICAL_TABLES = ("CONTRACT_FILES", "FROZEN_ATOM_DIGESTS", "FROZEN_REGION_DIGESTS", "OP_MANIFEST_REGISTRY_FILES")
ATOM_ID = re.compile(r"(?:04|05)-[A-Z]+-[1-9][0-9]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
REGISTRY_REF = re.compile(r"(?<![A-Za-z0-9_/])(?:spec/)?registry/[A-Za-z0-9._/-]+\.md")
DOCTRINE = (
    "Guard artifact: changing a guard merely to accept an edit is not a repair. "
    "A changed atom or region owes its owning spec/design update, every consuming "
    "contract, and an adversarial mutation. Required identities, anchors and "
    "Frozen-contract-change acknowledgements remain authoritative."
)


def path_value(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ValueError(f"invalid contract path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or path.suffix != ".md":
        raise ValueError(f"invalid contract path: {value!r}")
    return value


def read_inventory(source: str) -> dict:
    """Read literal declarations; never import a historical oracle."""
    try:
        module = ast.parse(source)
        parents = {child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)}
        names = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
        current = bool(names & set(TABLES[1:3]))
        if current and names & set(HISTORICAL_TABLES[1:3]):
            raise ValueError("mixed current and historical Phase 4B inventories")
        tables = TABLES if current else HISTORICAL_TABLES
        result = {}
        declarations = {}
        for name in tables:
            nodes = [node for node in module.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
            if len(nodes) != 1 or len(nodes[0].targets) != 1:
                raise ValueError(f"{name} requires one literal declaration")
            declarations[name] = nodes[0].targets[0]
            value = nodes[0].value
            for dictionary in (node for node in ast.walk(value) if isinstance(node, ast.Dict)):
                keys = [ast.literal_eval(key) for key in dictionary.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError(f"duplicate inventory key in {name}")
            result[name] = ast.literal_eval(value)
        for node in ast.walk(module):
            if isinstance(node, ast.Name) and node.id in tables and isinstance(node.ctx, (ast.Store, ast.Del)):
                if node is not declarations[node.id]:
                    raise ValueError(f"{node.id} cannot be rebound")
            if isinstance(node, ast.Name) and node.id in tables[1:] and isinstance(node.ctx, ast.Load):
                parent = parents[node]
                readonly_subscript = isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Load)
                call = parents.get(parent)
                readonly_method = isinstance(parent, ast.Attribute) and parent.attr in {"get", "items", "keys", "values"} and isinstance(call, ast.Call) and call.func is parent
                immutable_atoms = current and node.id == "REQUIRED_ATOMS"
                if not (immutable_atoms or readonly_subscript or readonly_method):
                    raise ValueError(f"{node.id} requires direct read-only access; aliases are unsupported")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if isinstance(receiver, ast.Name) and receiver.id in tables and node.func.attr not in {"get", "items", "keys", "values"}:
                    raise ValueError(f"{receiver.id} must remain a literal inventory")
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in tables and not isinstance(node.ctx, ast.Load):
                    raise ValueError(f"{node.value.id} cannot be mutated")
    except (SyntaxError, TypeError) as error:
        raise ValueError(f"invalid Phase 4B inventory: {error}") from error
    files = result["CONTRACT_FILES"]
    if not isinstance(files, tuple) or not files or not all(isinstance(path, str) for path in files) or len(files) != len(set(files)):
        raise ValueError("CONTRACT_FILES must be a nonempty unique literal tuple")
    for path in files:
        path_value(path)
    if not set((*ATOM_FILES, BUILTINS)) <= set(files):
        raise ValueError("the report requires both atom chapters and the builtin registry")
    atom_name, region_name, registry_name = tables[1:]
    atoms, regions, registries = (result[name] for name in tables[1:])
    for name, value in ((region_name, regions), (registry_name, registries)):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a literal dictionary")
    if current:
        if not isinstance(atoms, tuple) or not all(isinstance(atom, str) for atom in atoms) or len(atoms) != len(set(atoms)):
            raise ValueError("REQUIRED_ATOMS must be a unique literal tuple")
    else:
        if not isinstance(atoms, dict):
            raise ValueError(f"{atom_name} must be a literal dictionary")
        for atom, digest in atoms.items():
            if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
                raise ValueError(f"invalid historical atom digest: {atom!r}")
        atoms = tuple(atoms)
    for atom in atoms:
        if not isinstance(atom, str) or not ATOM_ID.fullmatch(atom):
            raise ValueError(f"invalid required atom: {atom!r}")
    boundaries = {}
    for label, row in regions.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(row, tuple) or len(row) != (3 if current else 4):
            raise ValueError(f"invalid region: {label!r}")
        path, start, end = row[:3]
        if path_value(path) not in files or not all(isinstance(marker, str) and marker for marker in (start, end)) or start == end:
            raise ValueError(f"invalid region boundaries: {label}")
        if not current and (not isinstance(row[3], str) or not DIGEST.fullmatch(row[3])):
            raise ValueError(f"invalid historical region digest: {label}")
        boundaries[label] = (path, start, end)
    for atom, path in registries.items():
        if not isinstance(atom, str) or not ATOM_ID.fullmatch(atom) or path_value(path) not in files or not path.startswith("spec/registry/"):
            raise ValueError(f"invalid registry owner: {atom!r}")
    return {"CONTRACT_FILES": files, "REQUIRED_ATOMS": atoms,
            "REQUIRED_REGIONS": boundaries, "OP_MANIFEST_REGISTRY_FILES": registries}


def location(text: str, start: int, block: str) -> list[int]:
    return [text.count("\n", 0, start) + 1, text.count("\n", 0, start + len(block) - 1) + 1]


def snapshot(read: Callable[[str], str]) -> dict:
    inventory = read_inventory(read(ORACLE))
    docs = {path: read(path) for path in inventory["CONTRACT_FILES"]}
    entries = {}
    try:
        for path in ATOM_FILES:
            for match in ATOM_START.finditer(docs[path]):
                atom = match[1]
                if not ATOM_ID.fullmatch(atom) or atom[:2] != PurePosixPath(path).name[:2]:
                    raise ValueError(f"unexpected atom {atom} in {path}")
                block = strict_atom_block(docs[path], atom)
                entries[("atom", atom)] = {
                    "path": path, "lines": location(docs[path], match.start(), block),
                    "text": normalize_frozen_block(block), "registries": {},
                    "protected": atom in inventory["REQUIRED_ATOMS"],
                }
                for reference in REGISTRY_REF.findall(block):
                    registry = reference if reference.startswith("spec/") else f"spec/{reference}"
                    if path_value(registry) not in docs:
                        raise ValueError(f"atom {atom} names an undeclared registry {registry}")
                    entries[("atom", atom)]["registries"][registry] = normalize_frozen_block(docs[registry])
        for atom in inventory["REQUIRED_ATOMS"]:
            if ("atom", atom) not in entries:
                raise ValueError(f"missing frozen atom {atom}")
        for atom, path in inventory["OP_MANIFEST_REGISTRY_FILES"].items():
            if ("atom", atom) not in entries:
                raise ValueError(f"registry {path} names missing atom {atom}")
            entries[("atom", atom)]["registries"][path] = normalize_frozen_block(docs[path])
        builtin_rows = parse_registry(docs[BUILTINS])
        shared_text = "\n".join(line for line in docs[BUILTINS].splitlines() if not ROW.fullmatch(line))
        for bracketed in set(builtin_rows.values()):
            atom = bracketed[1:-1]
            if ("atom", atom) not in entries:
                raise ValueError(f"builtin registry names missing atom {atom}")
            identities = sorted(identity for identity, owner in builtin_rows.items() if owner == bracketed)
            entries[("atom", atom)]["registries"][BUILTINS] = normalize_frozen_block(shared_text) + json.dumps(identities)
        for label, (path, start, end) in inventory["REQUIRED_REGIONS"].items():
            block = frozen_region(docs[path], start, end, label)
            entries[("region", label)] = {
                "path": path, "lines": location(docs[path], docs[path].index(start), block),
                "text": normalize_frozen_block(block), "markers": [start, end],
            }
    except OracleError as error:
        raise ValueError(str(error)) from error
    return {"entries": entries, "documents": docs}


def compare_snapshots(before: dict, after: dict) -> list[dict]:
    changes = []
    for kind, identity in sorted(before["entries"].keys() | after["entries"].keys()):
        old, new = (side["entries"].get((kind, identity)) for side in (before, after))
        reasons = []
        if old is None:
            reasons.append("added")
        elif new is None:
            reasons.append("removed")
        else:
            for field, label in [("path", "source moved"), ("text", "definition changed"),
                                 ("registries", "registry changed"), ("markers", "boundaries changed"),
                                 ("protected", "protection changed")]:
                if old.get(field) != new.get(field):
                    reasons.append(label)
        if reasons:
            def describe(entry):
                if entry is None:
                    return None
                return {key: sorted(value) if key == "registries" else value
                        for key, value in entry.items() if key != "text"}
            changes.append({"kind": kind, "identity": identity, "changes": reasons,
                            "before": describe(old), "after": describe(new)})
    return changes


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        detail = error.stderr.decode(errors="replace").strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ValueError(f"cannot read committed Phase 4B evidence: {detail}") from error


def commit(repo: Path, reference: str) -> str:
    value = git(repo, "rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}").strip()
    if not SHA.fullmatch(value):
        raise ValueError(f"invalid commit: {reference!r}")
    return value


def resolve_comparison(repo: Path, candidate: str, base: str, pr_head: str) -> tuple[str, str]:
    if bool(base) == bool(pr_head):
        raise ValueError("provide exactly one nonempty --base or --pr-head")
    candidate = commit(repo, candidate)
    if pr_head:
        if not SHA.fullmatch(pr_head):
            raise ValueError("--pr-head must be the exact event head SHA")
        return resolve_pr_commits(repo, candidate, pr_head)
    return commit(repo, base), candidate


def acknowledgement_line(kind: str, identity: str) -> str:
    if kind == "atom":
        value = f"atom:{identity}"
    elif kind == "region":
        value = "region:" + json.dumps(identity, ensure_ascii=False)
    elif kind == "file":
        value = identity
    else:
        raise ValueError(f"unknown acknowledgement kind {kind!r}")
    return f"Frozen-contract-change: {value}"


def identity_acknowledgement_violations(result: dict, named: list[tuple[str, str]]) -> list[str]:
    required = {(row["kind"], row["identity"]) for row in result["changes"]}
    counts = Counter(named)
    violations = []
    for key, count in sorted(counts.items()):
        if count > 1:
            violations.append(f"duplicate contract identity acknowledgement: {acknowledgement_line(*key)}")
        if key not in required:
            violations.append(f"stale or unknown contract identity acknowledgement: {acknowledgement_line(*key)}")
    for key in sorted(required - counts.keys()):
        violations.append(f"unacknowledged contract identity: add {acknowledgement_line(*key)}; {DOCTRINE}")
    return violations


def changed_contracts(repo: Path, base_ref: str, candidate_ref: str = "HEAD") -> dict:
    comparison, candidate = commit(repo, base_ref), commit(repo, candidate_ref)
    bases = git(repo, "merge-base", "--all", comparison, candidate).splitlines()
    if len(bases) != 1:
        raise ValueError("Phase 4B comparison requires one unique merge base")
    base = bases[0]
    before, after = (snapshot(lambda path, revision=revision: git(repo, "show", f"{revision}:{path}")) for revision in (base, candidate))
    files = sorted(before["documents"].keys() | after["documents"].keys())
    changes = compare_snapshots(before, after)
    changed_files = [path for path in files if before["documents"].get(path) != after["documents"].get(path)]
    return {
        "version": 1, "comparison_ref": comparison, "base": base, "candidate": candidate,
        "changes": changes,
        "changed_contract_files": changed_files,
        "required_acknowledgements": [acknowledgement_line("file", path) for path in changed_files]
            + [acknowledgement_line(row["kind"], row["identity"]) for row in changes],
        "doctrine": DOCTRINE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="", help="base comparison ref for a push or local run")
    parser.add_argument("--pr-head", default="", help="exact PR event head; validate the synthetic candidate's parents")
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.output.unlink(missing_ok=True)
        base, candidate = resolve_comparison(ROOT, args.candidate, args.base, args.pr_head)
        result = changed_contracts(ROOT, base, candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, OSError) as error:
        print(f"PHASE 4B CHANGE REPORT: FAIL: {error}", file=sys.stderr)
        return 1
    print(f'Phase 4B comparison: {result["base"]} -> {result["candidate"]}')
    for row in result["changes"]:
        print(f'  {row["kind"]} {row["identity"]}: {", ".join(row["changes"])}')
    for path in result["changed_contract_files"]:
        print(f"  changed contract file: {path}")
    print(DOCTRINE)
    print(f'PHASE 4B CHANGE REPORT: PASS ({len(result["changes"])} changed identities; contract review required)')
    return 0


if __name__ == "__main__":
    sys.exit(main())
