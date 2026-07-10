#!/usr/bin/env python3
"""chelis#434: propose a draft special-function envelope for {exp, log, sqrt}.

The FLOAT-PROPOSER half of the generalized envelope recipe (the exp/log/sqrt
counterpart of ``scripts/generate_erf_envelope.py``). Reads a config from
``crates/chelis-prove/data/special_fn_envelopes/<fn>.config.json`` (box
decomposition + per-box degree), fits a Chebyshev-node least-squares polynomial
per central box against an mpmath high-precision truth, and emits a DRAFT
``SpecialFnEnvelope`` JSON with ``eps`` unset (``0.0``).

The draft is NOT sound on its own: the Rust ``certify_special_fn_envelope stamp``
bin (``--features arb``) stamps each box's rigorous ``eps`` via the Arb whole-box
certifier. Soundness rests entirely on that certify step; a poor fit here only
yields a larger (still sound) ``eps``.

Coeffs are emitted as exact C99 hex-float strings (``float.hex()``), which the
Rust hex-float codec parses bit-exactly — so the certifier stamps an ``eps`` for
the same f64 polynomial the committed envelope carries.

Pipeline:
  1. .venv/bin/python scripts/generate_special_fn_envelope.py exp > /tmp/exp_draft.json
  2. cargo run -p chelis-prove --features arb --bin certify_special_fn_envelope \
       -- stamp /tmp/exp_draft.json \
       crates/chelis-prove/data/special_fn_envelopes/exp_envelope.json
  3. git diff crates/chelis-prove/data/special_fn_envelopes/

Exit codes: 0 draft written; 2 missing dependency / bad function / bad config.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "crates" / "chelis-prove" / "data" / "special_fn_envelopes"
FIT_NODES = 512
FIT_DPS = 50


def _truth(mpmath, fn: str):
    return {"exp": mpmath.exp, "log": mpmath.log, "sqrt": mpmath.sqrt}[fn]


def central_coeffs(fn: str, lo: float, hi: float, degree: int) -> list[float]:
    import mpmath
    import numpy as np

    mpmath.mp.dps = FIT_DPS
    truth = _truth(mpmath, fn)
    k = np.arange(FIT_NODES)
    cheb = np.cos((2 * k + 1) * np.pi / (2 * FIT_NODES))
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    xs = mid + half * cheb
    ys = np.array([float(truth(mpmath.mpf(float(x)))) for x in xs])
    coeffs = np.polyfit(xs, ys, degree)  # descending degree, Horner-ready
    return [float(c) for c in coeffs]


def build_draft(fn: str) -> dict:
    cfg_path = CONFIG_DIR / f"{fn}.config.json"
    if not cfg_path.exists():
        sys.stderr.write(f"error: no config at {cfg_path}\n")
        raise SystemExit(2)
    cfg = json.loads(cfg_path.read_text())
    boxes = []
    for b in cfg["boxes"]:
        if b["arm"] != "central":
            sys.stderr.write(f"error: only central arms supported, got {b['arm']}\n")
            raise SystemExit(2)
        coeffs = central_coeffs(fn, float(b["lo"]), float(b["hi"]), int(b["degree"]))
        boxes.append(
            {
                "lo": float(b["lo"]),
                "hi": float(b["hi"]),
                "arm": {"kind": "central", "coeffs": [c.hex() for c in coeffs]},
                "eps": 0.0,  # DRAFT: the Arb certifier stamps the sound eps
                "proof_kind": "arb_enclosure",
            }
        )
    return {
        "fn_name": fn,
        "domain": cfg["domain"],
        "output_clamp": None,
        "boxes": boxes,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("exp", "log", "sqrt"):
        sys.stderr.write("usage: generate_special_fn_envelope.py {exp|log|sqrt}\n")
        return 2
    try:
        import mpmath  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        sys.stderr.write(f"error: missing dependency ({e}); need numpy + mpmath\n")
        return 2
    draft = build_draft(argv[1])
    print(json.dumps(draft, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
