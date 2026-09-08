#!/usr/bin/env python3
"""All-or-nothing pre-Phase-4C prerequisite execution framework (#1296).

The production manifest deliberately contains MissingOracle entries. Existing
child drivers also need framework-owned execution-receipt adapters before this
command can accept them. A neighboring green suite or a printed PASS line does
not complete those deliveries. See dtype_semantics.md, Pre-4C, for ownership.

Usage: .venv/bin/python scripts/dtype_pre_phase4c_oracle.py
Acceptance: exit zero and final line DTYPE PRE-PHASE-4C ORACLE: PASS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
PASS_LINE = "DTYPE PRE-PHASE-4C ORACLE: PASS"
REQUIRED_ISSUES = frozenset({
    722, 753, 759, 893, 965, 1059, 1281, 1282, 1284, 1287, 1288, 1289,
    1290, 1292, 1293, 1294, 1295, 1297, 1298, 1306, 1313,
})
# These exact prerequisites prove census and atom closure. They execute their
# guards but own no evaluator/backend behavior cells. No receipt can choose this
# disposition; every other owning issue requires all three host lanes.
STRUCTURAL_ISSUES = frozenset({1288, 1294})
_NAME = re.compile(r"[a-z][a-z0-9-]*")


class OracleFailure(RuntimeError):
    """A prerequisite or its current execution evidence is absent or invalid."""


@dataclass(frozen=True)
class ChildOracle:
    name: str
    issues: tuple[int, ...]
    argv: tuple[str, ...]
    success_line: str


@dataclass(frozen=True)
class MissingOracle:
    name: str
    issues: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class SourceIdentity:
    head: str
    digest: str


def prerequisites(python: str) -> tuple[ChildOracle | MissingOracle, ...]:
    """Exact reviewed command identities; file existence never grants authority."""
    return (
        ChildOracle("freeze", (), (python, "scripts/dtype_phase4b_oracle.py"),
                    "DTYPE PHASE 4B ORACLE: PASS"),
        MissingOracle("integer-unary", (722,), "complete compiled unary/AD oracle pending"),
        MissingOracle("wrapping", (753,), "complete exact-width modular oracle pending"),
        MissingOracle("lossy-casts", (759,), "complete named cast-domain oracle pending"),
        MissingOracle("representation", (893, 1289),
                      "required typed-carrier/access oracle pending; Phase 0 is detection only"),
        MissingOracle("classification", (965,), "complete stored-width classification oracle pending"),
        MissingOracle("host-rendering", (1059,), "complete compiled Tensor/List rendering oracle pending"),
        MissingOracle("reductions", (1281,), "complete reduction/window extrema oracle pending"),
        MissingOracle("to-string", (1282,), "complete recursive to_string domain oracle pending"),
        MissingOracle("nonnumeric-lowering", (1284,), "typed logical/comparison/where oracle pending"),
        ChildOracle("count", (1287,), (python, "scripts/dtype_count_oracle.py"),
                    "DTYPE COUNT ORACLE: PASS"),
        MissingOracle("capacity", (1288,), "zero-exception capacity oracle pending"),
        MissingOracle("balanced-reductions", (1290,), "canonical balanced reduction oracle pending"),
        MissingOracle("tensor-close", (1292,), "own-width tensor comparison oracle pending"),
        MissingOracle("stdlib", (1293,), "complete stdlib/JSON/adjoint oracle pending"),
        MissingOracle("builtin-closure", (1294,), "exact builtin-atom closure oracle not merged"),
        MissingOracle("random-rounding-padding", (1295,), "all-active-dtype behavior oracle pending"),
        MissingOracle("host-effects", (1297,), "compiled host-effect oracle pending"),
        MissingOracle("runtime-axes-windows", (1298,), "complete runtime-axis/window oracle pending"),
        ChildOracle("direct-arithmetic", (1306,),
                    (python, "scripts/dtype_direct_arithmetic_oracle.py"),
                    "DTYPE DIRECT ARITHMETIC ORACLE: PASS"),
        ChildOracle("relu", (1313,), (python, "scripts/dtype_relu_oracle.py"),
                    "DTYPE RELU ORACLE: PASS"),
    )


def validate_manifest(children: Sequence[ChildOracle | MissingOracle]) -> None:
    names: set[str] = set()
    owners: set[int] = set()
    commands: set[tuple[str, ...]] = set()
    for child in children:
        if not isinstance(child, (ChildOracle, MissingOracle)):
            raise OracleFailure("unknown prerequisite disposition")
        if not _NAME.fullmatch(child.name) or child.name in names:
            raise OracleFailure(f"invalid or duplicate child identity: {child.name!r}")
        names.add(child.name)
        if not child.issues and child.name != "freeze":
            raise OracleFailure(f"child {child.name} has no owning prerequisite")
        for issue in child.issues:
            if type(issue) is not int or issue in owners:
                raise OracleFailure(f"invalid or duplicate prerequisite owner #{issue}")
            owners.add(issue)
        if isinstance(child, ChildOracle):
            if (not child.argv or child.argv in commands or not child.success_line
                    or "\n" in child.success_line or child.success_line == PASS_LINE):
                raise OracleFailure(f"invalid or duplicate child command: {child.name}")
            commands.add(child.argv)
        elif not child.reason:
            raise OracleFailure(f"missing child {child.name} has no stated deliverable")
    if owners != REQUIRED_ISSUES or "freeze" not in names:
        raise OracleFailure(f"prerequisite coverage drift: missing={sorted(REQUIRED_ISSUES - owners)}, "
                            f"extra={sorted(owners - REQUIRED_ISSUES)}; freeze required")


def require_available(children: Sequence[ChildOracle | MissingOracle]) -> tuple[ChildOracle, ...]:
    missing = [child for child in children if isinstance(child, MissingOracle)]
    if missing:
        detail = "; ".join(f"{child.name} ({', '.join(f'#{i}' for i in child.issues)}): "
                           f"{child.reason}" for child in missing)
        raise OracleFailure(f"missing prerequisite oracles: {detail}")
    return tuple(child for child in children if isinstance(child, ChildOracle))


def _object(value: object, keys: set[str], context: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise OracleFailure(f"{context}: expected exactly {sorted(keys)}")
    return value


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OracleFailure(f"duplicate JSON receipt field {key!r}")
        result[key] = value
    return result


def _identities(value: object, context: str, *, allow_empty: bool = False) -> set[str]:
    if (not isinstance(value, list) or (not value and not allow_empty)
            or any(not isinstance(item, str) or not item for item in value)):
        raise OracleFailure(f"{context}: expected nonempty test identities")
    if len(set(value)) != len(value):
        raise OracleFailure(f"{context}: duplicate identities")
    return set(value)


def validate_receipt(payload: object, child: ChildOracle, identity: SourceIdentity,
                     run_id: str) -> int:
    """Validate a child adapter's current, per-case execution packet.

    Adapters must obtain selections and outcomes from test-framework execution,
    including negative/mutation cases. This validator is not a substitute for
    implementing and adversarially validating that adapter in each child oracle.
    """
    p = _object(payload, {"schema", "run_id", "head", "source_digest", "oracle", "argv",
                          "selected", "executed", "obligations", "hosts", "devices"}, child.name)
    expected = {"schema": 1, "run_id": run_id, "head": identity.head,
                "source_digest": identity.digest, "oracle": child.name, "argv": list(child.argv)}
    for key, value in expected.items():
        if type(p[key]) is not type(value) or p[key] != value:
            raise OracleFailure(f"{child.name}: stale or incorrect receipt {key}")
    selected = _identities(p["selected"], f"{child.name} selection")
    if not isinstance(p["executed"], list):
        raise OracleFailure(f"{child.name}: missing per-test execution receipts")
    observed: set[str] = set()
    for row in p["executed"]:
        row = _object(row, {"id", "outcome"}, "test execution")
        if not isinstance(row["id"], str) or row["id"] in observed:
            raise OracleFailure(f"{child.name}: invalid or duplicate executed test")
        if row["outcome"] != "passed":
            raise OracleFailure(f"{child.name}: {row['id']} outcome {row['outcome']!r}")
        observed.add(row["id"])
    if observed != selected:
        raise OracleFailure(f"{child.name}: selected/executed mismatch: "
                            f"missing={sorted(selected - observed)}, extra={sorted(observed - selected)}")
    obligations = _object(p["obligations"], {"positive", "negative", "mutation"}, "obligations")
    for kind, cases in obligations.items():
        required = _identities(cases, f"{child.name} {kind}")
        if not required.issubset(observed):
            raise OracleFailure(f"{child.name}: unexecuted {kind} obligations")
    if set(child.issues) - STRUCTURAL_ISSUES:
        hosts = _object(p["hosts"], {"eval", "c-host", "c-dag"}, "host execution")
        for lane, cases in hosts.items():
            if not _identities(cases, f"{child.name} {lane}").issubset(observed):
                raise OracleFailure(f"{child.name}: unexecuted host lane {lane}")
    elif p["hosts"] != {}:
        raise OracleFailure(f"{child.name}: structural oracle declares host execution")
    if not isinstance(p["devices"], list):
        raise OracleFailure(f"{child.name}: device dispositions must be an explicit list")
    device_cells: set[tuple[str, str]] = set()
    for row in p["devices"]:
        row = _object(row, {"lane", "cell", "kind", "issue"}, "unbuilt device cell")
        if (row["lane"] not in ("hip", "metal") or not isinstance(row["cell"], str)
                or not row["cell"] or row["kind"] != "Unimplemented"
                or type(row["issue"]) is not int or row["issue"] <= 0):
            raise OracleFailure(f"{child.name}: invalid device disposition")
        key = row["lane"], row["cell"]
        if key in device_cells:
            raise OracleFailure(f"{child.name}: duplicate device cell {key}")
        device_cells.add(key)
    return len(observed)


def source_identity(root: Path) -> SourceIdentity:
    def git(*args: str) -> bytes:
        result = subprocess.run(("git", *args), cwd=root, capture_output=True, check=False)
        if result.returncode:
            raise OracleFailure(f"cannot inspect source identity: {result.stderr.decode(errors='replace')}")
        return result.stdout
    head = git("rev-parse", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise OracleFailure("cannot resolve exact source head")
    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise OracleFailure("authoritative composite requires a clean committed worktree")
    return SourceIdentity(head, hashlib.sha256(git("ls-files", "--stage", "-z")).hexdigest())


def run_child(child: ChildOracle, root: Path, identity: SourceIdentity, run_id: str,
              output: Path, *, extra_env: Mapping[str, str] | None = None) -> dict:
    """Execute one exact child command, preserving its private evidence directory."""
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise OracleFailure(f"{child.name}: cannot create fresh evidence directory: {error}") from error
    receipt_path = output / "execution.json"
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    env.update({"CHELIS_ORACLE_RECEIPT": str(receipt_path.resolve()),
                "CHELIS_ORACLE_RUN_ID": run_id, "CHELIS_ORACLE_HEAD": identity.head,
                "CHELIS_ORACLE_SOURCE_DIGEST": identity.digest})
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            completed = subprocess.run(child.argv, cwd=root, env=env, stdout=stdout,
                                       stderr=stderr, check=False)
    except OSError as error:
        raise OracleFailure(f"{child.name}: could not execute exact command: {error}") from error
    stdout_text = (output / "stdout.log").read_text()
    if completed.returncode:
        raise OracleFailure(f"{child.name}: exit {completed.returncode}; logs: {output}")
    lines = stdout_text.splitlines()
    if not lines or lines[-1] != child.success_line or lines.count(child.success_line) != 1:
        raise OracleFailure(f"{child.name}: missing or duplicate final success line; logs: {output}")
    try:
        packet = json.loads(receipt_path.read_text(), object_pairs_hook=_unique_json_fields)
    except (OSError, ValueError) as error:
        raise OracleFailure(f"{child.name}: missing/invalid execution receipt; "
                            f"child receipt adapter required: {error}") from error
    count = validate_receipt(packet, child, identity, run_id)
    return {"oracle": child.name, "issues": list(child.issues), "argv": list(child.argv),
            "success_line": child.success_line, "executed_tests": count,
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "receipt": str(receipt_path.relative_to(root))}


def validate_device_authorities(receipts: Sequence[dict], root: Path) -> None:
    """Use the existing issue-kind/open-state authority; never accept a waiver."""
    from capacity_census_liveness import IssueKind, IssueState, fetch_issue
    from generate_rejection_registries import MANIFEST_REL, load_issue_manifest

    registered = set(load_issue_manifest(root / MANIFEST_REL))
    cited: set[int] = set()
    for receipt in receipts:
        packet = json.loads((root / receipt["receipt"]).read_text(),
                            object_pairs_hook=_unique_json_fields)
        cited.update(row["issue"] for row in packet["devices"])
    for issue in sorted(cited):
        if issue not in registered:
            raise OracleFailure(f"unregistered unbuilt-device authority #{issue}")
        record = fetch_issue(issue)
        if record is None or record.kind is not IssueKind.ISSUE or record.state is not IssueState.OPEN:
            raise OracleFailure(f"unbuilt-device authority #{issue} is not a resolvable OPEN issue")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path,
                        help="new directory under this worktree's target/ for execution evidence")
    args = parser.parse_args(argv)
    try:
        manifest = prerequisites(sys.executable)
        validate_manifest(manifest)
        children = require_available(manifest)
        identity = source_identity(REPO_ROOT)
        run_id = uuid.uuid4().hex
        output = (args.receipt_dir or REPO_ROOT / "target/pre-phase4c" / run_id).resolve()
        if not output.is_relative_to(REPO_ROOT / "target") or output.exists():
            raise OracleFailure("receipt directory must be new and under this worktree's target/")
        receipts = []
        for child in children:
            print(f"running {child.name}: {' '.join(child.argv)}", flush=True)
            receipts.append(run_child(child, REPO_ROOT, identity, run_id, output / child.name))
        validate_device_authorities(receipts, REPO_ROOT)
        if source_identity(REPO_ROOT) != identity:
            raise OracleFailure("source changed during prerequisite execution")
        summary = {"schema": 1, "head": identity.head, "source_digest": identity.digest,
                   "run_id": run_id, "children": receipts}
        (output / "receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(PASS_LINE)
        return 0
    except (OracleFailure, OSError, ValueError) as error:
        print(f"DTYPE PRE-PHASE-4C ORACLE: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
