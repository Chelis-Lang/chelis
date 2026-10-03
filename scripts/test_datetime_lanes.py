#!/usr/bin/env python3
"""Unit tests for the shared Std.Datetime differential lanes (chelis#2942).

A stub `chelis` and a stub compiler record their arguments, so the tests
check what each lane runs, in which order, and which stage a failure lands
in, without building anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime_lanes as lanes  # noqa: E402

# `chelis --version` prints a version; `eval` prints a binding; `build --emit-c`
# writes the C sources and the staged archive. FAIL_EVAL fails the eval and
# FAIL_BUILD the build. Each call also logs the lane environment it saw.
CHELIS = """\
import json, os, pathlib, sys
log = pathlib.Path(os.environ["LANES_LOG"])
log.open("a").write(json.dumps(["chelis", *sys.argv[1:]]) + "\\n")
seen = {name: os.environ.get(name) for name in ("CHELIS_REEF_HOME", "CHELIS_STYLE_GATE_DISABLE", "OMP_NUM_THREADS")}
pathlib.Path(os.environ["LANES_LOG"] + ".env").open("a").write(json.dumps(seen) + "\\n")
if sys.argv[1:] == ["--version"]:
    print("chelis 9.9.9")
elif sys.argv[1] == "eval" and os.environ.get("FAIL_EVAL"):
    print("error: date: domain: x", file=sys.stderr)
    sys.exit(2)
elif sys.argv[1] == "eval":
    print("x = [1]")
elif os.environ.get("FAIL_BUILD"):
    print("error: build failed", file=sys.stderr)
    sys.exit(3)
else:
    out = pathlib.Path("out")
    out.mkdir()
    (out / "main.c").write_text("int main(void) { return 0; }\\n")
    (out / "libchelis_runtime.a").write_text("")
"""

# Writes `-o`'s target as a script that prints a binding, or traps with status
# 134 under FAIL_RUN; FAIL_LINK fails the link.
COMPILER = """\
import json, os, pathlib, sys
log = pathlib.Path(os.environ["LANES_LOG"])
log.open("a").write(json.dumps(["cc", *sys.argv[1:]]) + "\\n")
if os.environ.get("FAIL_LINK"):
    print("ld: undefined symbol", file=sys.stderr)
    sys.exit(1)
target = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
target.write_text("#!/bin/sh\\nif [ -n \\"$FAIL_RUN\\" ]; then echo 'overflow in add' >&2; exit 134; fi\\necho 'x = [1]'\\n")
target.chmod(0o755)
"""


def stub(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


class LaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.log = root / "log.jsonl"
        self.env = {"LANES_LOG": str(self.log)}
        self.patch = mock.patch.dict(os.environ, self.env)
        self.patch.start()
        chelis = stub(root, "chelis", CHELIS)
        compiler = stub(root, "cc", COMPILER)
        toolchain = lanes.Toolchain.from_json(json.dumps(
            {"compiler": str(compiler), "compile_flags": ["-O1", "-Werror"], "link_flags": ["-lm"]}))
        self.runner = lanes.Runner(chelis, root / "reef", root / "work", toolchain, 60, "lanes-test")

    def tearDown(self) -> None:
        self.patch.stop()
        self.tmp.cleanup()

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()][1:]  # drop --version

    def test_c_lane_emits_c_then_compiles_once_and_runs(self) -> None:
        result = self.runner.lane("case_1", "module Demo.Main\n", "c")
        self.assertEqual((result.lane, result.status, result.stage, result.stdout), ("c", 0, "run", "x = [1]\n"))
        self.assertGreaterEqual(result.seconds, 0.0)
        self.assertEqual(self.calls(), [
            ["chelis", "build", "--emit-c", "src/main.ch", "--target", "c", "--output", "out"],
            ["cc", "-O1", "-Werror", "-Iout", "out/main.c", "out/libchelis_runtime.a", "-lm", "-o", "out/case"],
        ])
        app = Path(self.tmp.name) / "work" / "c" / "case_1"
        self.assertEqual((app / "src" / "main.ch").read_text(), "module Demo.Main\n")
        self.assertIn('name = "lanes-test"', (app / "reef.toml").read_text())
        self.assertIn('compiler = "=9.9.9"', (app / "reef.toml").read_text())

    def test_eval_lane_runs_chelis_eval(self) -> None:
        result = self.runner.lane("case_1", "module Demo.Main\n", "eval")
        self.assertEqual((result.lane, result.status, result.stage, result.stdout), ("eval", 0, "eval", "x = [1]\n"))
        self.assertEqual(self.calls(), [["chelis", "eval", "--file", "src/main.ch"]])

    def test_a_failed_build_stops_before_the_compiler(self) -> None:
        with mock.patch.dict(os.environ, {"FAIL_BUILD": "1"}):
            result = self.runner.lane("case_1", "module Demo.Main\n", "c")
        self.assertEqual((result.status, result.stage), (3, "build"))
        self.assertIn("build failed", result.stderr)
        self.assertEqual([call[0] for call in self.calls()], ["chelis"])

    def test_a_failed_link_is_the_link_stage(self) -> None:
        with mock.patch.dict(os.environ, {"FAIL_LINK": "1"}):
            result = self.runner.lane("case_1", "module Demo.Main\n", "c")
        self.assertEqual((result.status, result.stage), (1, "link"))
        self.assertEqual([call[0] for call in self.calls()], ["chelis", "cc"])

    def test_a_timeout_is_the_timeout_stage(self) -> None:
        def expire(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 60)

        with mock.patch.object(self.runner, "run", expire):
            result = self.runner.lane("case_1", "module Demo.Main\n", "eval")
        self.assertEqual((result.status, result.stage, result.stderr), (None, "timeout", "timed out after 60 s"))

    def test_a_failing_executable_keeps_its_status_and_stderr(self) -> None:
        with mock.patch.dict(os.environ, {"FAIL_RUN": "1"}):
            result = self.runner.lane("case_1", "module Demo.Main\n", "c")
        self.assertEqual((result.status, result.stage, result.stderr), (134, "run", "overflow in add\n"))

    def test_a_failing_eval_keeps_its_status_and_stderr(self) -> None:
        with mock.patch.dict(os.environ, {"FAIL_EVAL": "1"}):
            result = self.runner.lane("case_1", "module Demo.Main\n", "eval")
        self.assertEqual((result.status, result.stage, result.stderr), (2, "eval", "error: date: domain: x\n"))

    def test_every_lane_process_gets_the_reef_home_and_the_timeout(self) -> None:
        with mock.patch.object(lanes.subprocess, "run", wraps=subprocess.run) as run:
            self.runner.lane("case_1", "module Demo.Main\n", "eval")
            self.runner.lane("case_1", "module Demo.Main\n", "c")
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all(call.kwargs["timeout"] == 60 for call in run.call_args_list))
        lane_envs = [json.loads(line) for line in Path(f"{self.log}.env").read_text().splitlines()][1:]
        self.assertEqual(lane_envs, [{
            "CHELIS_REEF_HOME": str(Path(self.tmp.name) / "reef"),
            "CHELIS_STYLE_GATE_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
        }] * 2)

    def test_a_rerun_starts_from_a_fresh_package(self) -> None:
        stale = Path(self.tmp.name) / "work" / "eval" / "case_1" / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("left over")
        self.runner.lane("case_1", "module Demo.Main\n", "eval")
        self.assertFalse(stale.exists())

    def test_the_c_lane_needs_a_toolchain(self) -> None:
        self.runner.toolchain = None
        with self.assertRaisesRegex(ValueError, "needs a toolchain"):
            self.runner.lane("case_1", "module Demo.Main\n", "c")


if __name__ == "__main__":
    unittest.main()
