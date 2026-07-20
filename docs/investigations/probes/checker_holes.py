#!/usr/bin/env python3
"""Which wrapper constructs hide an ill-typed body from chelis check? (#709 sweep)"""

import json
import os
import subprocess
from pathlib import Path

CHELIS = os.environ.get("CHELIS_BIN", "target/debug/chelis")
BAD = "add(cast(1.0, f32), cast(2, int64))"
DIR = Path(__file__).parent / "probes" / "checker"
DIR.mkdir(parents=True, exist_ok=True)

CASES = {
    "bare": f"def f() -> f32 = {BAD}\n",
    "with_seed": f"def f() -> f32 = with seed(42i64) {{ {BAD} }}\n",
    "with_device": f'def f() -> f32 = with device("gpu:0") {{ {BAD} }}\n',
    "let_body": f"def f() -> f32 = let x = {BAD} in x\n",
    "if_then": f"def f(c: bool) -> f32 = if c then {BAD} else 1.0\n",
    "lambda_body": f"def f() -> f32 = (fn (v: f32) -> f32 = {BAD})(1.0)\n",
    "pipe_stage": f"def f() -> f32 = 1.0 |> fn (v: f32) -> f32 = {BAD}\n",
    "tuple_elem": f"def f() -> (f32, f32) = ({BAD}, 1.0)\n",
    "list_elem": f"def f() -> [f32] = [{BAD}, 1.0]\n",
    "match_arm": f"def f(c: bool) -> f32 = match c {{ true => {BAD}, false => 1.0 }}\n",
    "grad_body": f"def g(x: f32) -> f32 = {BAD}\ndef f(x: f32) -> f32 = grad(g)(x)\n",
    "vmap_lambda": f"def f(t: tensor[4, f32]) -> tensor[4, f32] = vmap(fn (v: f32) -> f32 = {BAD})(t)\n",
    "jit_body": f"def g(x: f32) -> f32 = {BAD}\ndef f(x: f32) -> f32 = jit(g)(x)\n",
    "nested_with_seed": f"def f() -> f32 = with seed(42i64) {{ with seed(7i64) {{ {BAD} }} }}\n",
    "seed_expr_ill_typed": 'def f() -> f32 = with seed("not a seed") { 1.0 }\n',
    "device_expr_ill_typed": "def f() -> f32 = with device(42) { 1.0 }\n",
}

for name, prog in CASES.items():
    p = DIR / f"{name}.ch"
    p.write_text(prog)
    r = subprocess.run(
        [CHELIS, "check", str(p)],
        capture_output=True,
        text=True,
        env={"CHELIS_STYLE_GATE_DISABLE": "1", "PATH": "/usr/bin:/bin"},
    )
    try:
        score = json.loads(r.stdout).get("score")
    except Exception:
        score = f"NO_JSON exit={r.returncode} err={r.stderr.strip().splitlines()[:1]}"
    flag = "  <-- HOLE" if score == 1 else ""
    print(f"{name:24} score={score}{flag}")
