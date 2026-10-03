"""Run one Chelis program on the eval and compiled-C lanes (chelis#2942).

The Std.Datetime differentials (`datetime_differential.py`,
`datetime_business_differential.py`, `datetime_zone_differential.py`) share
this lane code. Each program is a one-file reef package under the runner's
work directory. The eval lane runs `chelis eval --file`. The C lane runs
`chelis build --emit-c`, which writes the C sources and stages the runtime
archive without compiling them, then compiles and links them once under the
strict reference toolchain, and runs the executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


@dataclass(frozen=True)
class Toolchain:
    compiler: str
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]

    @classmethod
    def from_json(cls, text: str) -> Toolchain:
        """The `--toolchain-json` argument: `{compiler, compile_flags, link_flags}`."""
        spec = json.loads(text)
        return cls(spec["compiler"], tuple(spec["compile_flags"]), tuple(spec["link_flags"]))


@dataclass
class LaneResult:
    lane: str
    status: int | None
    stdout: str
    stderr: str
    stage: str
    seconds: float = 0.0


class Runner:
    def __init__(self, chelis: Path, reef_home: Path, work: Path, toolchain: Toolchain | None, timeout: int,
                 package: str):
        self.chelis = chelis
        self.reef_home = reef_home
        self.work = work
        self.toolchain = toolchain
        self.timeout = timeout
        version = subprocess.run([str(chelis), "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
        self.manifest = (f'schema = "1"\n\n[package]\nname = "{package}"\nversion = "0.1.0"\n'
                         f'compiler = "={version}"\nmodule_prefix = "Demo"\n\n[dependencies]\nchelis-std = {{ version = "0.4.0" }}\n')

    def app(self, name: str, source: str, lane: str) -> Path:
        app = self.work / lane / name
        if app.exists():
            shutil.rmtree(app)
        (app / "src").mkdir(parents=True)
        (app / "reef.toml").write_text(self.manifest, encoding="utf-8")
        (app / "src" / "main.ch").write_text(source, encoding="utf-8")
        return app

    def run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update({"CHELIS_REEF_HOME": str(self.reef_home), "CHELIS_STYLE_GATE_DISABLE": "1", "OMP_NUM_THREADS": "1"})
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=self.timeout, check=False)

    def lane(self, name: str, source: str, lane: str) -> LaneResult:
        """Run program `name` with `source` on `lane`, timed; a process past the timeout is stage `timeout`."""
        started = time.monotonic()
        try:
            result = self.eval_lane(name, source) if lane == "eval" else self.c_lane(name, source)
        except subprocess.TimeoutExpired as error:
            result = LaneResult(lane, None, "", f"timed out after {error.timeout} s", "timeout")
        result.seconds = time.monotonic() - started
        return result

    def eval_lane(self, name: str, source: str) -> LaneResult:
        app = self.app(name, source, "eval")
        done = self.run([str(self.chelis), "eval", "--file", "src/main.ch"], app)
        return LaneResult("eval", done.returncode, done.stdout, done.stderr, "eval")

    def c_lane(self, name: str, source: str) -> LaneResult:
        if self.toolchain is None:
            raise ValueError("the c lane needs a toolchain")
        app = self.app(name, source, "c")
        build = self.run([str(self.chelis), "build", "--emit-c", "src/main.ch", "--target", "c", "--output", "out"], app)
        if build.returncode != 0:
            return LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        link = self.run([self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", "out/main.c",
                         "out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"], app)
        if link.returncode != 0:
            return LaneResult("c", link.returncode, link.stdout, link.stderr, "link")
        done = self.run([str(app / "out" / "case")], app)
        return LaneResult("c", done.returncode, done.stdout, done.stderr, "run")
