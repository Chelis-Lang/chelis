#!/usr/bin/env python3
"""Tests for scripts/generate_special_fn_envelope.py (the chelis#434 exp/log/sqrt
float proposer).

Run with the uv-managed Python:
  .venv/bin/python -m pytest scripts/test_generate_special_fn_envelope.py

Covers the DRAFT structure and the fit-quality contract. Not soundness — that is
the Arb certifier's job (`certify_special_fn_envelope`); a poor fit only yields a
larger certified eps, never an unsound envelope.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

try:
    import numpy as np

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import pytest
except ImportError:
    # pytest lives only in the uv venv; provide a minimal stub so unittest
    # discovery imports this module without error.
    class _Stub:
        class mark:  # noqa: N801
            @staticmethod
            def skipif(*_a, **_k):
                return lambda f: f

        @staticmethod
        def skip(*_a, **_k):
            return None

    pytest = _Stub()  # type: ignore

MOD_PATH = Path(__file__).resolve().parent / "generate_special_fn_envelope.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_special_fn", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRUTH = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt}


@pytest.mark.skipif(not HAVE_DEPS, reason="needs numpy + mpmath")
def test_draft_structure_and_fit_for_each_function():
    mod = _load()
    for fn in ("exp", "log", "sqrt"):
        draft = mod.build_draft(fn)
        assert draft["fn_name"] == fn
        assert draft["boxes"], f"{fn} has boxes"
        for b in draft["boxes"]:
            assert b["proof_kind"] == "arb_enclosure"
            assert b["eps"] == 0.0, "draft eps is unset (certifier stamps it)"
            coeffs = b["arm"]["coeffs"]
            assert all(isinstance(c, str) and c.startswith(("0x", "-0x")) for c in coeffs), (
                f"{fn} coeffs must be hex-float strings"
            )
            # Coeffs parse back to a finite f64 and the fit is non-garbage. This
            # is a STRUCTURAL sanity check, not a tight-fit claim: the certifier
            # makes the envelope sound regardless, and log/sqrt boxes that reach
            # near their domain edge (0) fit worse (see README). exp fits tightly.
            cs = [float.fromhex(c) for c in coeffs]
            assert all(math.isfinite(c) for c in cs), f"{fn} coeffs finite"
            xs = np.linspace(b["lo"], b["hi"], 2001)
            err = float(np.max(np.abs(np.polyval(cs, xs) - np.vectorize(TRUTH[fn])(xs))))
            threshold = 1e-3 if fn == "exp" else 5e-2
            assert err < threshold, f"{fn} box [{b['lo']},{b['hi']}] fit too poor: {err}"


@pytest.mark.skipif(not HAVE_DEPS, reason="needs numpy + mpmath")
def test_domain_matches_config():
    mod = _load()
    expected = {"exp": "all_reals", "log": "positive", "sqrt": "non_negative"}
    for fn, dom in expected.items():
        assert mod.build_draft(fn)["domain"] == dom


if __name__ == "__main__":
    if HAVE_DEPS:
        test_draft_structure_and_fit_for_each_function()
        test_domain_matches_config()
        print("ok")
    else:
        print("skipped (no numpy/mpmath)")
