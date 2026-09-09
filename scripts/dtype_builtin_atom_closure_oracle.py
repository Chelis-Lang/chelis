#!/usr/bin/env python3
"""Authoritative #1294 declaration/semantic-membership closure (not backend support).

Build discovery from Rust, execute the named Rust tests with nextest receipts,
then run per-identity and per-atom adversarial checks through unittest. A PASS
requires a clean, unchanged committed source and actual execution of all cases.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import unittest
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import builtin_atom_registry as registry  # noqa: E402
from scripts import builtin_atom_semantic_contracts as semantics  # noqa: E402

PASS_LINE = "DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS"
PROFILE = "builtin-atom-closure"
FILTER = ('binary(builtin_atom_discovery) + binary(issue_1294_normalize) '
          '+ test(every_risc_op_has_an_exact_pre_phase4c_atom_disposition)')
CARGO_TESTS = ["-p", "chelis-types", "-p", "chelis-ir", "-p", "chelis-cli", "--lib", "--test",
               "builtin_atom_discovery", "--test", "issue_1294_normalize",
               "-E", FILTER]


class OracleError(RuntimeError):
    pass


def command(argv: list[str], root: Path = ROOT) -> bytes:
    completed = subprocess.run(argv, cwd=root, stdout=subprocess.PIPE, check=False)
    if completed.returncode:
        raise OracleError(f"command failed ({completed.returncode}): {argv}")
    return completed.stdout


def source_identity(root: Path) -> tuple[str, str]:
    """Schema-1 source identity shared with #1296, computed from actual bytes."""
    head = command(["git", "rev-parse", "HEAD"], root).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise OracleError("cannot resolve exact HEAD")
    if command(["git", "status", "--porcelain", "--untracked-files=normal"], root).strip():
        raise OracleError("authoritative oracle requires a clean committed worktree")
    digest = hashlib.sha256()
    for record in command(["git", "ls-tree", "-r", "-z", head], root).split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, expected_blob = metadata.split()
        path = root / os.fsdecode(name)
        actual_mode = path.lstat().st_mode
        if mode == b"120000" and stat.S_ISLNK(actual_mode):
            data = os.fsencode(os.readlink(path))
        elif mode in (b"100644", b"100755") and stat.S_ISREG(actual_mode):
            if bool(actual_mode & 0o111) != (mode == b"100755"):
                raise OracleError(f"tracked source mode differs: {name!r}")
            data = path.read_bytes()
        else:
            raise OracleError(f"unsupported or changed tracked source kind: {name!r}")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest().encode()
        if kind != b"blob" or blob != expected_blob:
            raise OracleError(f"tracked source bytes differ from HEAD: {name!r}")
        digest.update(record + b"\0" + len(data).to_bytes(8, "big") + data)
    return head, digest.hexdigest()


def discover(packet: object) -> list[str]:
    if not isinstance(packet, dict) or set(packet) != {"builtins", "risc"}:
        raise OracleError("invalid compiled inventory packet")
    union = set()
    for source, rows in packet.items():
        if not isinstance(rows, list) or not rows or any(not isinstance(x, str) for x in rows):
            raise OracleError(f"empty or invalid {source} discovery")
        if len(set(rows)) != len(rows):
            raise OracleError(f"duplicate {source} discovery before union")
        union.update(rows)
    return sorted(union)


def nextest_selection(packet: dict) -> list[str]:
    selected = []
    for binary, suite in packet["rust-suites"].items():
        for name, case in suite["testcases"].items():
            if case["filter-match"]["status"] != "matches":
                continue
            if case["ignored"]:
                raise OracleError(f"ignored selected test {binary}::{name}")
            selected.append(f"rust:{binary}::{name}")
    if not selected or len(set(selected)) != len(selected):
        raise OracleError("empty or duplicate nextest selection")
    return sorted(selected)


def junit_execution(path: Path, selected: list[str]) -> list[dict[str, str]]:
    executed = []
    for suite in ET.parse(path).getroot().iter("testsuite"):
        for case in suite.findall("testcase"):
            if any(case.find(tag) is not None for tag in ("failure", "error", "skipped", "rerunFailure")):
                raise OracleError(f"non-passing nextest case {case.attrib}")
            executed.append({"id": f"rust:{suite.attrib['name']}::{case.attrib['name']}",
                             "outcome": "passed"})
    verify_execution(selected, executed)
    return executed


def verify_execution(selected: list[str], executed: list[dict[str, str]]) -> None:
    ids = [row["id"] for row in executed]
    if (not selected or len(set(selected)) != len(selected) or len(set(ids)) != len(ids)
            or set(selected) != set(ids) or any(row["outcome"] != "passed" for row in executed)):
        raise OracleError("selected/executed mismatch, duplicate, empty, or non-passing test")


class Case(unittest.TestCase):
    def __init__(self, identity, kind, action):
        super().__init__()
        self.identity, self.kind, self.action = identity, kind, action

    def id(self):
        return self.identity

    def runTest(self):
        self.action()


class Receipts(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.executed.append({"id": test.id(), "outcome": "passed"})


def closure_cases(identities: list[str], table: str, spec: str, generated: set[str]) -> list[Case]:
    rows = registry.validate(identities, table, spec, generated)
    semantics.validate_semantics(rows, spec)
    blocks = registry.atom_blocks(spec)
    cases = []

    def add(name, kind, action):
        cases.append(Case("closure:" + name, kind, action))

    def rejects(**changes):
        args = dict(discovered=identities, table=table, spec=spec, generated=generated)
        args.update(changes)
        try:
            changed_rows = registry.validate(**args)
            semantics.validate_semantics(changed_rows, args["spec"])
        except registry.RegistryError:
            return
        raise AssertionError("mutation was accepted")

    add("exact-bijection", "positive", lambda: registry.validate(identities, table, spec, generated))
    for identity, atom in rows.items():
        line = f"| `{identity}` | {atom} |"
        add(f"missing:{identity}", "mutation", lambda line=line: rejects(table=table.replace(line, "")))
        add(f"duplicate:{identity}", "mutation", lambda line=line: rejects(table=table + "\n" + line))
        add(f"stale:{identity}", "mutation", lambda line=line, identity=identity:
            rejects(table=table.replace(line, line.replace(identity, identity + "Stale"))))
        # Use a real existing atom from a disjoint operation family. A syntactically
        # valid atom and generated membership alone must not confer authority.
        other = next(a for a, b in blocks.items() if a != atom and
                     identity.split(":")[1] not in re.findall(r"`([A-Za-z][A-Za-z0-9_]*)(?:`|\()", b))
        add(f"wrong-authority:{identity}", "mutation", lambda line=line, atom=atom, other=other:
            rejects(table=table.replace(line, line.replace(atom, other))))
    for atom in sorted(set(rows.values())):
        block = blocks[atom]
        add(f"deleted-atom:{atom}", "mutation", lambda block=block: rejects(spec=spec.replace(block, "")))
        add(f"stale-generated:{atom}", "mutation", lambda atom=atom: rejects(generated=generated - {atom}))
        for field in registry.FIELDS:
            changed = block.replace(field + ":", "Removed:")
            add(f"deleted-{field}:{atom}", "mutation", lambda block=block, changed=changed:
                rejects(spec=spec.replace(block, changed)))
    for number, clauses in semantics.CLAUSES.items():
        atom = f"[05-OP-{number}]"
        block = blocks[atom]
        for index, clause in enumerate(clauses):
            # Collapse whitespace while preserving a real normative definition;
            # only the named semantic obligation is removed by this mutation.
            changed = "> " + semantics.normalized(block).replace(clause, "REMOVED") + "\n"
            add(f"semantic-clause:{atom}:{index}", "mutation", lambda block=block, changed=changed:
                rejects(spec=spec.replace(block, changed)))
    for identity in semantics.CASE_CLAUSES:
        atom = rows[identity]
        sibling = next(other for name, other in rows.items()
                       if name.split(":")[1] == identity.split(":")[1] and other != atom)
        line = f"| `{identity}` | {atom} |"
        add(f"wrong-overload:{identity}", "mutation", lambda line=line, atom=atom, sibling=sibling:
            rejects(table=table.replace(line, line.replace(atom, sibling))))
    for name, kwargs in [
        ("empty-discovery", {"discovered": []}),
        ("duplicate-discovery", {"discovered": identities + identities[:1]}),
        ("issue-is-not-authority", {"table": table.replace(next(iter(rows.values())), "chelis#1294", 1)}),
        ("observation-is-not-operation", {"table": table.replace(next(iter(rows.values())), "[05-OBS-1]", 1)}),
        ("nonexistent-atom", {"table": table.replace(next(iter(rows.values())), "[05-OP-999999]", 1)}),
    ]:
        add(name, "negative", lambda kwargs=kwargs: rejects(**kwargs))
    return cases


def main() -> None:
    # Delete an old output before any possible failure. No earlier PASS packet
    # can stand in for a run that failed its source or selection preflight.
    output = Path(os.environ.get("CHELIS_ORACLE_RECEIPT", ROOT / "target/builtin-atom-closure/execution.json"))
    output.unlink(missing_ok=True)
    identity = source_identity(ROOT)
    receipt_keys = ("CHELIS_ORACLE_RECEIPT", "CHELIS_ORACLE_RUN_ID", "CHELIS_ORACLE_HEAD", "CHELIS_ORACLE_SOURCE_DIGEST")
    if any(key in os.environ for key in receipt_keys):
        if not all(os.environ.get(key) for key in receipt_keys):
            raise OracleError("incomplete composite receipt handoff")
        if identity != (os.environ[receipt_keys[2]], os.environ[receipt_keys[3]]):
            raise OracleError("composite source identity differs from current source")
    inventory = json.loads(command(["cargo", "run", "--quiet", "--locked", "-p", "chelis-ir",
                                    "--example", "builtin_atom_inventory"]))
    identities = discover(inventory)
    command([sys.executable, "scripts/generate_rejection_registries.py", "--check"])
    spec = (ROOT / "spec/05-risc-primitives.md").read_text()
    table = (ROOT / "spec/registry/builtin_semantic_identities.md").read_text()
    generated = set(re.findall(r'"(\[05-OP-[1-9][0-9]*\])"',
        (ROOT / "crates/chelis-types/src/rejection_registry_generated.rs").read_text()))
    cases = closure_cases(identities, table, spec, generated)
    unit_suite = unittest.defaultTestLoader.loadTestsFromNames([
        "scripts.test_builtin_atom_registry", "scripts.test_dtype_builtin_atom_closure_oracle"])
    def flatten(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item
    unit_tests = list(flatten(unit_suite))
    python_selection = [case.id() for case in [*cases, *unit_tests]]
    result = unittest.TextTestRunner(verbosity=1, resultclass=Receipts).run(unittest.TestSuite([*cases, *unit_tests]))
    if not result.wasSuccessful():
        raise OracleError("Python closure or adversarial execution failed")
    verify_execution(python_selection, result.executed)
    selection = nextest_selection(json.loads(command([
        "cargo", "nextest", "list", "--locked", "--profile", PROFILE,
        "--ignore-default-filter", "--message-format", "json", *CARGO_TESTS])))
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    junit = target / "nextest" / PROFILE / "junit.xml"
    junit.unlink(missing_ok=True)
    command(["cargo", "nextest", "run", "--locked", "--profile", PROFILE,
             "--ignore-default-filter", "--no-fail-fast", "--retries", "0", *CARGO_TESTS])
    executed = result.executed + junit_execution(junit, selection)
    selected = python_selection + selection
    verify_execution(selected, executed)
    if source_identity(ROOT) != identity:
        raise OracleError("source changed during execution")
    packet = {"schema": 1, "run_id": os.environ.get("CHELIS_ORACLE_RUN_ID", str(uuid.uuid4())),
              "head": identity[0], "source_digest": identity[1], "oracle": "builtin-closure",
              "argv": [sys.executable, "scripts/dtype_builtin_atom_closure_oracle.py"],
              "selected": selected, "executed": executed,
              "obligations": {kind: [case.id() for case in cases if case.kind == kind]
                              for kind in ("positive", "negative", "mutation")},
              "hosts": {}, "devices": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n")
    print(f"{len(identities)} exact identities; {len(selected)} selected and passed tests; head {identity[0]}")
    print(PASS_LINE)


if __name__ == "__main__":
    try:
        main()
    except (OracleError, registry.RegistryError, OSError, ValueError, KeyError, StopIteration) as error:
        print(f"DTYPE BUILTIN ATOM CLOSURE ORACLE: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
