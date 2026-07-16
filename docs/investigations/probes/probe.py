#!/usr/bin/env python3
"""Lane-probe driver for the chelis numeric audit sweeps.

Usage:
  probe.py eval  <file.ch>          -> run chelis eval, print stdout/stderr/exit
  probe.py check <file.ch>          -> run chelis check
  probe.py c     <file.ch>          -> build --target c, link via printed Compile: line, run
  probe.py hip   <file.ch>          -> build --target hip, dump kernel names
  probe.py metal <file.ch>          -> build --target metal, dump kernel names

All lanes print verbatim strings; nothing is parsed through float.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

CHELIS = os.environ.get("CHELIS_BIN", "target/debug/chelis")
ENV = {"CHELIS_STYLE_GATE_DISABLE": "1", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, **kw)


def lane_eval(path):
    r = run([CHELIS, "eval", "--file", str(path)])
    return r.returncode, r.stdout, r.stderr


def lane_check(path):
    r = run([CHELIS, "check", str(path)])
    return r.returncode, r.stdout, r.stderr


def lane_build(path, target):
    out_dir = Path(path).with_suffix("") .as_posix() + f"-{target}-out"
    r = run([CHELIS, "build", str(path), "--target", target, "--output", out_dir])
    return r, Path(out_dir)


def lane_c(path):
    r, out_dir = lane_build(path, "c")
    if r.returncode != 0:
        return ("BUILD_FAIL", r.stdout + r.stderr, None)
    steps = re.findall(r"^Compile(?: object)?: (.+)$", r.stdout, re.M)
    if not steps:
        return ("NO_COMPILE_LINE", r.stdout + r.stderr, None)
    for step in steps:
        cc = run(shlex.split(step))
        if cc.returncode != 0:
            return ("LINK_FAIL", cc.stderr, None)
    binary = out_dir / Path(path).stem
    if not binary.exists():
        return ("NO_BINARY (library-only emission)", r.stdout, None)
    rr = run([str(binary)])
    emitted = (out_dir / (Path(path).stem + ".c")).read_text()
    return (f"RUN_EXIT={rr.returncode}", rr.stdout + rr.stderr, emitted)


def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "eval":
        code, out, err = lane_eval(path)
        print(f"EXIT={code}")
        print(out, end="")
        if err.strip():
            print("STDERR:", err.strip())
    elif mode == "check":
        code, out, err = lane_check(path)
        print(f"EXIT={code}")
        print(out, end="")
        if err.strip():
            print("STDERR:", err.strip())
    elif mode == "c":
        status, out, emitted = lane_c(path)
        print(status)
        print(out, end="")
    elif mode in ("hip", "metal"):
        r, out_dir = lane_build(path, mode)
        print(f"BUILD_EXIT={r.returncode}")
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                text = f.read_text(errors="replace")
                kernels = sorted(set(re.findall(r"kernel_[A-Za-z0-9_]+", text)))
                print(f"--- {f.name} ({len(text)} bytes) kernels={kernels}")
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
