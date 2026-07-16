#!/usr/bin/env python3
"""Run a battery of (name, program-kind, expr/program) probes through eval and C lanes.

Each probe row prints: name | eval result | C result. Results are verbatim
strings (never parsed through float). Errors are shown as ERR:<first line>.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe import lane_eval, lane_c  # noqa: E402

SCRATCH = Path(__file__).parent / "probes"
SCRATCH.mkdir(exist_ok=True)


def short(s, n=90):
    s = s.strip().replace("\n", " | ")
    return s[:n]


def eval_expr(name, expr):
    p = SCRATCH / f"bat_{name}_e.ch"
    p.write_text(f"module M.Main\nout = print({expr})\n")
    code, out, err = lane_eval(p)
    if code != 0:
        return "ERR: " + short(err.strip().splitlines()[-1] if err.strip() else out)
    return short(out.splitlines()[0] if out else "")


def eval_program(name, program):
    p = SCRATCH / f"bat_{name}_e.ch"
    p.write_text(program)
    code, out, err = lane_eval(p)
    if code != 0:
        return "ERR: " + short(err.strip().splitlines()[-1] if err.strip() else out)
    return short(" ; ".join(out.splitlines()))


def c_expr(name, expr, ret_ty):
    p = SCRATCH / f"bat_{name}_c.ch"
    p.write_text(f"def run() -> {ret_ty} = {expr}\nout = run()\n")
    status, out, _ = lane_c(p)
    if status.startswith("RUN_EXIT=0"):
        line = next((l for l in out.splitlines() if l.strip().startswith("out")), out)
        return short(line.split("=", 1)[-1])
    if status.startswith("RUN_EXIT"):
        return f"{status}: " + short(out)
    return f"{status}: " + short(out.strip().splitlines()[-1] if out.strip() else "")


def c_program(name, program):
    p = SCRATCH / f"bat_{name}_c.ch"
    p.write_text(program)
    status, out, _ = lane_c(p)
    if status.startswith("RUN_EXIT=0"):
        return short(" ; ".join(out.splitlines()))
    if status.startswith("RUN_EXIT"):
        return f"{status}: " + short(out)
    return f"{status}: " + short(out.strip().splitlines()[-1] if out.strip() else "")


def run_rows(rows):
    for row in rows:
        if len(row) == 3:
            name, expr, ret_ty = row
            e = eval_expr(name, expr)
            c = c_expr(name, expr, ret_ty)
        else:
            name, prog = row
            e = eval_program(name, prog)
            c = c_program(name, prog)
        flag = "AGREE" if e == c else "DIVERGE"
        print(f"{name:42} | eval: {e:44} | C: {c:44} | {flag}")


if __name__ == "__main__":
    import importlib

    mod = importlib.import_module(sys.argv[1])
    run_rows(mod.ROWS)
