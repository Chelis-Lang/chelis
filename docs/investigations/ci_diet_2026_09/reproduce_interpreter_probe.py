"""Time a bare Chelis JSON program, `check` and `eval` separately.

Written for chelis#1829, to answer whether the slowdown reaches a user program
rather than only a test harness, and which phase carries it. It showed `check`
flat across the regression (2.11 / 2.10 / 2.19 s) while `eval` went 1.80 s to
over 300 s, which is what moved the #1362 row from inferred to measured.

Invoke from a worktree checked out at the commit under test, after building the
binary it times (`cargo build -p chelis-cli --bin chelis`):

    .venv/bin/python bare_1829.py --label culprit-22b193cf7 --cap 300

Prints one JSON record with `check_s` and `eval_s` timed separately.

USE THIS TO VALIDATE A FIX, not the test suite. Because `check` is flat and
all of the cost is in `eval`, a candidate repair can be accepted or rejected
from this script alone in roughly five seconds per commit at the good end,
without building or running `chelis-cli`'s integration tests at all. The
affected test rows take 45+ minutes each while the defect is live, so
reaching for them first costs an hour to learn what this answers in seconds.

THREE PROPERTIES THAT ARE NOT OBVIOUS, and that a reimplementation will get
wrong by default:

1. `check` and `eval` are timed as SEPARATE invocations. Timing only the
   end-to-end run cannot tell a front-end regression from an interpreter one,
   and for chelis#1829 that distinction is the entire finding.

2. Everything is fresh per measurement: a new reef home, a freshly published
   chelis-std, a new package directory. `~/.cache/chelis` is workstation-wide and
   setting `XDG_CACHE_HOME` or `CHELIS_HOME` does NOT isolate it, so a reused
   package directory silently serves a result computed at a different commit.
   The program also needs a real package directory with a `reef.lock` present;
   a bare `chelis eval` cannot reach the stdlib at all.

3. Each invocation runs in its own process group, killed as a group on timeout,
   with a reap by target path afterwards - same reason as in `measure_1829.py`.
   `chelis` children outlive a killed parent and a survivor corrupts the next
   measurement while looking like noise.

Everything is per-measurement fresh: a new reef home, a freshly published
chelis-std and a new package directory, because `~/.cache/chelis` is
workstation-wide and will otherwise serve a result computed at a different
commit. Each invocation is capped and the target directory is reaped, so a
pathological run cannot outlive its step.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

WORKTREE = Path(
    os.environ.get("CHELIS_1829_WORKTREE", "/Users/robertronan/chelis-worktrees/1829-bisect")
)
TARGET = WORKTREE / "target"
CHELIS = TARGET / "debug" / "chelis"

PROGRAM = """module Demo.Main
import Std.Io.Json (JsonFloat, JsonObject, json_float, json_get, json_object, parse_json, to_json)
doc = parse_json(read_file("{input}"))
portfolio = match json_object(json_get(doc, "portfolio")) with {{
  | Some(entries) => JsonObject(entries)
  | None => fail("required JSON object missing at key `portfolio`")
}}
value = match json_float(json_get(portfolio, "rate")) with {{
  | Some(number) => number
  | None => fail("required JSON number missing at key `rate`")
}}
done = write_file("{output}", to_json(JsonFloat(value)))
"""
INPUT_JSON = '{"portfolio": {"rate": 0.0425}}\n'


def reap() -> list[int]:
    listing = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True).stdout
    killed = []
    for line in listing.splitlines():
        pid_text, _, args = line.strip().partition(" ")
        if str(TARGET) not in args:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    return killed


def timed(argv, cwd, env, cap):
    started = time.monotonic()
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        out = proc.communicate(timeout=cap)[0]
        return proc.returncode, round(time.monotonic() - started, 2), out
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return 124, round(time.monotonic() - started, 2), ""


parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--cap", type=int, default=300)
parser.add_argument(
    "--scratch",
    default=os.path.join(tempfile.gettempdir(), "chelis-1829-bare"),
    help="root for the per-measurement package directories; each --label gets a "
    "fresh subdirectory, which is what defeats the workstation-wide cache",
)
args = parser.parse_args()

root = Path(args.scratch) / args.label
if root.exists():
    shutil.rmtree(root)
(root / "app" / "src").mkdir(parents=True)
reef_home = root / "reef_home"
reef_home.mkdir()

version = re.search(
    r'(?m)^version = "([^"]+)"', (WORKTREE / "Cargo.toml").read_text()
).group(1)

env = dict(os.environ)
env["CHELIS_REEF_HOME"] = str(reef_home)
env["CHELIS_STYLE_GATE_DISABLE"] = "1"

std_copy = root / "chelis-std"
shutil.copytree(WORKTREE / "packages" / "chelis-std", std_copy)
code, publish_s, out = timed([str(CHELIS), "reef", "publish", str(std_copy)], root, env, args.cap)
if code != 0:
    print(json.dumps({"label": args.label, "stage": "publish", "code": code, "tail": out[-800:]}))
    raise SystemExit(2)

app = root / "app"
(app / "reef.toml").write_text(
    f'[package]\nname = "bare-json"\nversion = "0.1.0"\ncompiler = "={version}"\n'
    f'module_prefix = "Demo"\n\n[dependencies]\nchelis-std = {{ version = "0.4.0" }}\n'
)
(app / "input.json").write_text(INPUT_JSON)
source = app / "src" / "main.ch"
source.write_text(PROGRAM.format(input=app / "input.json", output=app / "output.json"))

check_code, check_s, check_out = timed(
    [str(CHELIS), "check", str(source)], app, env, args.cap
)
eval_code, eval_s, eval_out = timed(
    [str(CHELIS), "eval", "--file", str(source)], app, env, args.cap
)
print(json.dumps({
    "label": args.label, "version": version,
    "publish_s": publish_s,
    "check_s": check_s, "check_code": check_code,
    "eval_s": eval_s, "eval_code": eval_code,
    "eval_timed_out": eval_code == 124, "check_timed_out": check_code == 124,
    "lock": (app / "reef.lock").exists(),
    "reaped": reap(),
    "check_tail": check_out[-400:] if check_code not in (0, 124) else "",
    "eval_tail": eval_out[-400:] if eval_code not in (0, 124) else "",
}))
