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
import sysconfig
import tempfile
import time
import tomllib
import unittest

from capacity_census_graph import GraphError
import ci_timing


@dataclass(frozen=True)
class TestExecution:
    command: tuple[str, ...]
    selected: tuple[str, ...]
    executed: tuple[str, ...]
    output_sha256: str
    wire_probe_target: str | None = None
    wire_probe_sha256: str | None = None


WIRE_PROBE_TARGET_ENV = "CHELIS_WIRE_SELECTED_PROBE_TARGET"
WIRE_PROBE_SHA256_ENV = "CHELIS_WIRE_SELECTED_PROBE_SHA256"


def validate_wire_probe(root: Path, target: Path, expected_sha256: str) -> Path:
    """Bind a selected probe to this worktree's existing Cargo target and bytes."""
    root = root.resolve()
    target = Path(target)
    if (
        not target.is_absolute()
        or target.resolve() != target
        or not target.is_dir()
        or not target.is_relative_to(root / "target")
    ):
        raise GraphError("selected wire probe target is not owned by this worktree")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(byte not in "0123456789abcdef" for byte in expected_sha256)
    ):
        raise GraphError("selected wire probe lacks an exact binary digest")
    probe = target / "debug/examples/wire_publication_probe"
    if not probe.is_file() or probe.is_symlink() or probe.resolve() != probe:
        raise GraphError("selected wire publication probe is missing or redirected")
    if hashlib.sha256(probe.read_bytes()).hexdigest() != expected_sha256:
        raise GraphError("selected wire publication probe differs from schema evidence")
    return target


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
        self.started_at = {}

    def startTest(self, test):
        name = test.id()
        if name in self.started:
            raise GraphError("duplicate unittest start")
        self.started.add(name)
        super().startTest(test)
        self.started_at[name] = time.monotonic()
        ci_timing.record({"event": "start", "name": name, "kind": "census-control"})

    def stopTest(self, test):
        name = test.id()
        if name not in self.started or name in self.stopped:
            raise GraphError("unmatched unittest stop")
        self.stopped.add(name)
        ci_timing.record({
            "event": "finish",
            "name": name,
            "kind": "census-control",
            "seconds": time.monotonic() - self.started_at.pop(name),
            "outcome": self.outcomes.get(name, "unreported"),
        })
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


def _supervise(root, receipt_path, names, wire_probe=None):
    names = _selection(names)
    if wire_probe is not None:
        target, digest = wire_probe
        target = validate_wire_probe(root, target, digest)
        os.environ[WIRE_PROBE_TARGET_ENV] = str(target)
        os.environ[WIRE_PROBE_SHA256_ENV] = digest
    else:
        os.environ.pop(WIRE_PROBE_TARGET_ENV, None)
        os.environ.pop(WIRE_PROBE_SHA256_ENV, None)
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
        if wire_probe is not None:
            validate_wire_probe(root, target, digest)
        packet = {
            "schema": 1,
            "selected": list(names),
            "started": sorted(result.started),
            "stopped": sorted(result.stopped),
            "tests_run": result.testsRun,
            "outcomes": result.outcomes,
        }
        if wire_probe is not None:
            packet["wire_probe"] = {"target": str(target), "sha256": digest}
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


def _managed_python_environment():
    """Bind native builds and embedded Python to this interpreter's packages.

    PYO3_PYTHON selects libpython at build time, but a Rust executable does
    not discover a Python venv from VIRTUAL_ENV when it initializes Python.
    Supply that venv's site directories explicitly and exclude ambient ones.
    """
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONUSERBASE", None)
    environment.update({
        "PYO3_PYTHON": sys.executable,
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(
            sysconfig.get_path(key) for key in ("purelib", "platlib")
        )),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    })
    return environment


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
            **_managed_python_environment(),
            "RUSTC_BOOTSTRAP": "1",
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
    expected_kinds = [kind]
    if kind == "lib":
        try:
            manifest = tomllib.loads((source.parent.parent / "Cargo.toml").read_text())
            library = manifest.get("lib", {})
            if not isinstance(library, dict):
                raise ValueError("library declaration is not a table")
            expected_kinds = library.get("crate-type", ["lib"])
            if (not isinstance(expected_kinds, list) or not expected_kinds
                    or any(not isinstance(value, str) or value not in {
                        "lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro",
                    } for value in expected_kinds)
                    or len(set(expected_kinds)) != len(expected_kinds)):
                raise ValueError("library crate-type is not a concrete supported declaration")
        except (OSError, UnicodeError, ValueError) as error:
            raise GraphError("invalid current library manifest for test artifact selection") from error
    matches = [
        Path(item["executable"]).resolve()
        for item in artifacts
        if item.get("reason") == "compiler-artifact"
        and item.get("target", {}).get("name") == name
        and item["target"].get("kind") == expected_kinds
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
        **_managed_python_environment(),
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
        "RUSTC_BOOTSTRAP": "1",
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


def run_python_tests(
    root: Path, selected, *, log_prefix=None,
    wire_probe_target: Path | None = None, wire_probe_sha256: str | None = None,
) -> TestExecution:
    selected = _selection(selected)
    root = root.resolve()
    if (wire_probe_target is None) != (wire_probe_sha256 is None):
        raise GraphError("selected wire probe requires both target and digest")
    if wire_probe_target is not None:
        wire_probe_target = validate_wire_probe(
            root, wire_probe_target, wire_probe_sha256
        )
    with tempfile.TemporaryDirectory(prefix="chelis-wire-execution-") as directory:
        path = Path(directory) / "execution.json"
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "--supervise",
            str(root),
            str(path),
            json.dumps(selected),
            *(
                (
                    "--wire-probe-target", str(wire_probe_target),
                    "--wire-probe-sha256", wire_probe_sha256,
                )
                if wire_probe_target is not None else ()
            ),
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
        if wire_probe_target is not None:
            validate_wire_probe(root, wire_probe_target, wire_probe_sha256)
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
    if wire_probe_target is not None:
        expected["wire_probe"] = {
            "target": str(wire_probe_target), "sha256": wire_probe_sha256,
        }
    if packet != expected or type(packet.get("tests_run")) is not int:
        raise GraphError(
            "Python execution was missing, skipped, failed or not selected"
        )
    return TestExecution(
        command, selected, selected, hashlib.sha256(raw).hexdigest(),
        str(wire_probe_target) if wire_probe_target is not None else None,
        wire_probe_sha256,
    )


if __name__ == "__main__":
    try:
        if sys.argv[1:2] != ["--supervise"] or len(sys.argv) not in (5, 9):
            raise GraphError("only the wire verifier may invoke the receipt supervisor")
        wire_probe = None
        if len(sys.argv) == 9:
            if sys.argv[5] != "--wire-probe-target" or sys.argv[7] != "--wire-probe-sha256":
                raise GraphError("invalid selected wire probe arguments")
            wire_probe = (Path(sys.argv[6]), sys.argv[8])
        from ci_timing import subprocesses
        with subprocesses():
            raise SystemExit(
                _supervise(
                    Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4]),
                    wire_probe,
                )
            )
    except (GraphError, OSError, ValueError) as error:
        print(f"wire execution failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
