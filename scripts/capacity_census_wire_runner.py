"""Framework-owned execution receipts for the C6 wire acceptance command.

This runner records exact selected test identities, starts and outcomes. A
command's exit status or a test module's printed success line is insufficient.
Source/artifact binding and selection ownership remain with the wire verifier.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from capacity_census_graph import GraphError


@dataclass(frozen=True)
class TestExecution:
    command: tuple[str, ...]
    selected: tuple[str, ...]
    executed: tuple[str, ...]
    output_sha256: str


def _selection(names):
    names = tuple(names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise GraphError("test selection must be nonempty, exact and unique")
    return tuple(sorted(names))


def validate_libtest_execution(selected, events):
    """Check libtest JSON lifecycle events, including the framework totals."""
    selected = _selection(selected)
    started = set()
    completed = set()
    opened = False
    closed = False
    for event in events:
        if not isinstance(event, dict) or closed:
            raise GraphError("unexpected libtest event outside the selected suite")
        kind, action = event.get("type"), event.get("event")
        if kind == "suite" and action == "started" and not opened:
            count = event.get("test_count")
            if type(count) is not int or count != len(selected):
                raise GraphError("libtest suite selection count differs")
            opened = True
        elif kind == "test" and opened:
            name = event.get("name")
            if name not in selected:
                raise GraphError("libtest executed an unselected test")
            if action == "started" and name not in started:
                started.add(name)
            elif action == "ok" and name in started and name not in completed:
                completed.add(name)
            else:
                raise GraphError(
                    "libtest missing, duplicate, failed or skipped outcome"
                )
        elif kind == "suite" and action == "ok" and opened:
            if started != set(selected) or completed != set(selected):
                raise GraphError("libtest selected cases did not all execute")
            expected = {
                "passed": len(selected),
                "failed": 0,
                "ignored": 0,
                "measured": 0,
            }
            if any(
                type(event.get(key)) is not int or event[key] != count
                for key, count in expected.items()
            ):
                raise GraphError("libtest final totals do not match execution")
            if type(event.get("filtered_out")) is not int or event["filtered_out"] < 0:
                raise GraphError("libtest missing explicit excluded-test count")
            closed = True
        else:
            raise GraphError("unsupported or failed libtest lifecycle event")
    if not closed:
        raise GraphError("libtest did not finish the selected suite")
    return tuple(sorted(completed))


class _OwnedResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.started = set()
        self.stopped = set()
        self.outcomes = {}

    def startTest(self, test):
        name = test.id()
        if name in self.started:
            raise GraphError("duplicate unittest start")
        self.started.add(name)
        super().startTest(test)

    def stopTest(self, test):
        name = test.id()
        if name not in self.started or name in self.stopped:
            raise GraphError("unmatched unittest stop")
        self.stopped.add(name)
        super().stopTest(test)

    def _record(self, test, outcome):
        name = test.id()
        if name in self.outcomes:
            # In particular, a failed subtest cannot be cleared by a later
            # success callback on its parent method.
            self.outcomes[name] = "failed"
        else:
            self.outcomes[name] = outcome

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed")

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "failed")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped")

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "failed")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "failed")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(test, "failed")


def _tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _tests(item)
        elif isinstance(item, unittest.TestCase):
            yield item.id()
        else:
            raise GraphError("unsupported unittest selection item")


def _supervise(root, receipt_path, names):
    names = _selection(names)
    # Open the framework's receipt before importing the suite. An early exit
    # leaves no complete JSON packet, even when it exits with status zero.
    with receipt_path.open("x") as output:
        sys.path[:0] = [str(root), str(root / "scripts")]
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(names)
        if loader.errors or _selection(_tests(suite)) != names:
            raise GraphError("unittest loader did not select every declared obligation")
        result = _OwnedResult()
        suite.run(result)
        packet = {
            "schema": 1,
            "selected": list(names),
            "started": sorted(result.started),
            "stopped": sorted(result.stopped),
            "tests_run": result.testsRun,
            "outcomes": result.outcomes,
        }
        json.dump(packet, output, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    for test, trace in (*result.errors, *result.failures):
        print(f"{test}:\n{trace}", file=sys.stderr)
    return 0 if result.wasSuccessful() else 1


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GraphError("duplicate execution receipt field")
        result[key] = value
    return result


def run_libtest(root: Path, binary: Path, selected, *, log_prefix=None) -> TestExecution:
    selected = _selection(selected)
    binary = binary.resolve()
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    command = (
        str(binary),
        "-Zunstable-options",
        "--format=json",
        "--exact",
        "--test-threads=1",
        *selected,
    )
    result = subprocess.run(
        command,
        cwd=root,
        env={
            **os.environ,
            "RUSTC_BOOTSTRAP": "1",
            "PYO3_PYTHON": sys.executable,
            "VIRTUAL_ENV": sys.prefix,
        },
        capture_output=True,
        check=False,
    )
    if log_prefix is not None:
        from capacity_census_wire_calls import record_process
        record_process(log_prefix, command, result)
    try:
        events = [
            json.loads(line, object_pairs_hook=_unique_fields)
            for line in result.stdout.splitlines()
        ]
        executed = validate_libtest_execution(selected, events)
    except (ValueError, GraphError) as error:
        raise GraphError(
            f"invalid actual libtest execution: {error}\n"
            "stdout:\n" + result.stdout.decode(errors="replace")[-6000:]
            + "\nstderr:\n" + result.stderr.decode(errors="replace")[-6000:]
        ) from error
    if result.returncode or hashlib.sha256(binary.read_bytes()).hexdigest() != before:
        raise GraphError("libtest process failed or its executable changed")
    return TestExecution(
        command, selected, executed, hashlib.sha256(result.stdout).hexdigest()
    )


def select_test_binary(target: Path, source: Path, name: str, artifacts, *, kind="test") -> Path:
    if kind not in {"test", "lib"}:
        raise GraphError("unsupported Rust test artifact kind")
    matches = [
        Path(item["executable"]).resolve()
        for item in artifacts
        if item.get("reason") == "compiler-artifact"
        and item.get("target", {}).get("name") == name
        and item["target"].get("kind") == [kind]
        and item["target"].get("src_path") == str(source.resolve())
        and item.get("profile", {}).get("test") is True
        and item.get("executable")
    ]
    if len(matches) != 1 or not matches[0].is_relative_to(target.resolve()):
        raise GraphError(
            "Cargo did not emit one exact test artifact in the owned target"
        )
    return matches[0]


def build_and_run_rust_test(
    root: Path,
    target: Path,
    package: str,
    name: str,
    selected,
    *,
    kind="test",
    log_prefix=None,
):
    """Use Cargo's actual artifact identity; never glob for a warm executable."""
    root, target = root.resolve(), target.resolve()
    if not target.is_relative_to(root / "target"):
        raise GraphError("Rust test target must belong to this worktree")
    if kind == "lib":
        if name != package.replace("-", "_"):
            raise GraphError("library test name must match its exact crate")
        source = root / "crates" / package / "src/lib.rs"
        selection = ("--lib",)
    elif kind == "test":
        source = root / "crates" / package / "tests" / (name + ".rs")
        selection = ("--test", name)
    else:
        raise GraphError("unsupported Rust test artifact kind")
    if not source.is_file() or not source.resolve().is_relative_to(root / "crates"):
        raise GraphError("missing or foreign exact Rust test source")
    command = (
        "cargo",
        "test",
        "--locked",
        "--no-run",
        "--message-format=json",
        "-p",
        package,
        *selection,
    )
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
        "RUSTC_BOOTSTRAP": "1",
        "PYO3_PYTHON": sys.executable,
        "VIRTUAL_ENV": sys.prefix,
    }
    result = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, check=False
    )
    if log_prefix is not None:
        from capacity_census_wire_calls import record_process
        record_process(log_prefix.with_name(log_prefix.name + "-build"), command, result)
    if result.returncode:
        raise GraphError("wire consumer test build failed: " + result.stderr[-6000:])
    try:
        artifacts = [
            json.loads(line, object_pairs_hook=_unique_fields)
            for line in result.stdout.splitlines()
        ]
    except ValueError as error:
        raise GraphError("invalid Cargo JSON test artifact output") from error
    binary = select_test_binary(target, source, name, artifacts, kind=kind)
    return run_libtest(root, binary, selected, log_prefix=log_prefix)


def run_python_tests(root: Path, selected, *, log_prefix=None) -> TestExecution:
    selected = _selection(selected)
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="chelis-wire-execution-") as directory:
        path = Path(directory) / "execution.json"
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--supervise",
            str(root),
            str(path),
            json.dumps(selected),
        )
        result = subprocess.run(command, cwd=root, capture_output=True, check=False)
        if log_prefix is not None:
            from capacity_census_wire_calls import record_process
            record_process(log_prefix, command, result)
        if result.returncode:
            raise GraphError(
                "selected Python execution failed: "
                + result.stderr.decode(errors="replace")[-4000:]
            )
        try:
            raw = path.read_bytes()
            packet = json.loads(raw, object_pairs_hook=_unique_fields)
            if log_prefix is not None:
                log_prefix.with_suffix(".execution.json").write_bytes(raw)
        except (OSError, ValueError) as error:
            raise GraphError(
                "missing or invalid supervised Python execution receipt"
            ) from error
    expected = {
        "schema": 1,
        "selected": list(selected),
        "started": list(selected),
        "stopped": list(selected),
        "tests_run": len(selected),
        "outcomes": {name: "passed" for name in selected},
    }
    if packet != expected or type(packet.get("tests_run")) is not int:
        raise GraphError(
            "Python execution was missing, skipped, failed or not selected"
        )
    return TestExecution(command, selected, selected, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    try:
        if len(sys.argv) != 5 or sys.argv[1] != "--supervise":
            raise GraphError("only the wire verifier may invoke the receipt supervisor")
        raise SystemExit(
            _supervise(Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4]))
        )
    except (GraphError, OSError, ValueError) as error:
        print(f"wire execution failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
