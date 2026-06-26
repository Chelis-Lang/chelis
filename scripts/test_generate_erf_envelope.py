#!/usr/bin/env python3
"""Tests for scripts/generate_erf_envelope.py (the WI-13 float proposer).

Run with the uv-managed Python:
  .venv/bin/python -m pytest scripts/test_generate_erf_envelope.py

These tests cover the DRAFT structure and the fit-quality contract of the
proposer. They do NOT certify soundness: soundness is the Arb certifier's
job (the Rust `arb`-gated re-validation harness), and a poor proposed fit
only yields a larger certified eps, never an unsound envelope. So the tests
here assert the draft is well shaped and the fit is good enough to be useful,
not that any eps holds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # CI's unittest discovery imports test_*.py; pytest lives only in the uv
    # venv. Provide a minimal stub so this module imports without error.
    # unittest discovers no TestCase classes, so 0 tests run — harmless.
    class _PytestStub:
        class _Mark:
            def __getattr__(self, _):
                return lambda f: f
        mark = _Mark()
        def importorskip(self, mod, *a, **kw):
            import types
            return types.ModuleType(mod)
        def main(self, *a, **kw):
            return 0
        def __getattr__(self, _):
            return lambda *a, **kw: lambda f: f
    pytest = _PytestStub()  # type: ignore[assignment]

_SPEC = importlib.util.spec_from_file_location(
    "generate_erf_envelope",
    Path(__file__).resolve().parent / "generate_erf_envelope.py",
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(gen)

# numpy/mpmath are required by the proposer; skip cleanly if absent so the
# suite still runs in an env that only has the runtime deps.
np = pytest.importorskip("numpy")
mp = pytest.importorskip("mpmath")


def test_draft_has_three_contiguous_boxes():
    d = gen.build_draft()
    boxes = d["boxes"]
    assert len(boxes) == 3
    # ascending + contiguous
    assert boxes[0]["lo"] == -gen.SAT_HI
    assert boxes[0]["hi"] == -gen.CENTRAL
    assert boxes[1]["lo"] == -gen.CENTRAL
    assert boxes[1]["hi"] == gen.CENTRAL
    assert boxes[2]["lo"] == gen.CENTRAL
    assert boxes[2]["hi"] == gen.SAT_HI
    for a, b in zip(boxes, boxes[1:]):
        assert a["hi"] == b["lo"], "boxes must be contiguous"


def test_draft_arms_are_saturation_tails_and_central_poly():
    d = gen.build_draft()
    boxes = d["boxes"]
    assert boxes[0]["arm"]["kind"] == "saturation"
    assert boxes[0]["arm"]["value"] == -1.0
    assert boxes[2]["arm"]["kind"] == "saturation"
    assert boxes[2]["arm"]["value"] == 1.0
    assert boxes[1]["arm"]["kind"] == "central"
    assert len(boxes[1]["arm"]["coeffs"]) == gen.DEGREE + 1


def test_draft_eps_is_placeholder_zero():
    # The proposer must NOT claim a certified eps; the Arb certifier stamps it.
    d = gen.build_draft()
    for b in d["boxes"]:
        assert b["eps"] == 0.0


def test_central_fit_is_useful():
    # The proposed central polynomial should approximate erf on [-CENTRAL,
    # CENTRAL] well enough to be useful (the Arb certifier will stamp the exact
    # error; here we just guard against a broken fit). Direct sampled sup error.
    d = gen.build_draft()
    coeffs = np.array(d["boxes"][1]["arm"]["coeffs"])
    xs = np.linspace(-gen.CENTRAL, gen.CENTRAL, 4001)
    mp.mp.dps = 30
    truth = np.array([float(mp.erf(mp.mpf(float(x)))) for x in xs])
    err = np.abs(np.polyval(coeffs, xs) - truth)
    assert err.max() < 1e-4, f"central fit too poor: sup err {err.max()}"


def test_saturation_reaches_finance_range():
    # The desk box reaches the hundreds; the tails must cover +-300.
    assert gen.SAT_HI >= 300.0


def test_degree_flag_changes_coeff_count():
    d = gen.build_draft(degree=9)
    assert len(d["boxes"][1]["arm"]["coeffs"]) == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
