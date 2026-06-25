#!/usr/bin/env python3
"""WI-13 Option B: assemble the committed erf envelope from the Gappa proof bundle.

Builds the single committed envelope ``crates/chelis-prove/data/erf_envelope.json``
the runtime consumes, with ALL THREE arms sourced from the Gappa proof manifest
(``crates/chelis-prove/data/erf_proof/manifest.json``):

  - CENTRAL arm: the Sollya remez polynomial + its Gappa-proved sup-norm eps.
  - SATURATION tails (+-1): the Gappa-proved tail eps. The tails are proved by
    the SAME subdivision + certified-Taylor-model + Gappa machinery as the
    central arm -- the approximation is the constant +-1, so Gappa proves
    |+-1 - T(x)| <= bound (pure polynomial; it needs no erf/exp).

Every box's proof_kind is "gappa": each arm has a committed, machine-checkable
Gappa proof term an auditor re-runs. The Arb whole-box certifier remains the
independent every-build CROSS-CHECK (``certify_erf_envelope validate``), run
separately in the pipeline, asserting every committed eps also bounds the Arb
sup-norm.

Pipeline:
  1. .venv/bin/python scripts/generate_erf_proof.py        # Gappa bundle (3 arms)
  2. .venv/bin/python scripts/assemble_erf_envelope.py     # this script
  3. cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
       -- validate crates/chelis-prove/data/erf_envelope.json   # Arb cross-check
  4. git diff crates/chelis-prove/data/

Exit codes:
  0  envelope assembled and written
  2  the Gappa manifest is missing (run generate_erf_proof.py first)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "crates" / "chelis-prove" / "data"
PROOF_MANIFEST = DATA_DIR / "erf_proof" / "manifest.json"
ENVELOPE_OUT = DATA_DIR / "erf_envelope.json"

# Arb subdivision count recorded in provenance and used by the Arb cross-check
# (`certify_erf_envelope validate`). Fine enough that the Arb sup-norm bound on
# the central box drops just below the Gappa-proved eps, so the cross-check
# (committed eps >= Arb bound) passes with the committed eps = the Gappa value.
ARB_SUBDIVISIONS = 2097152


def build_envelope(manifest: dict) -> dict:
    central_lo = float(manifest["central_lo"])
    central_hi = float(manifest["central_hi"])
    sat_hi = float(manifest["sat_hi"])
    # Coeffs as JSON numbers (the manifest already stores the exact doubles as
    # numbers), so the envelope and manifest read back bit-identically.
    coeffs = [float(c) for c in manifest["coeffs"]]

    boxes = [
        {
            "lo": -sat_hi,
            "hi": central_lo,
            "arm": {"kind": "saturation", "value": -1.0},
            "eps": float(manifest["tail_neg_eps"]),
            "proof_kind": "gappa",
        },
        {
            "lo": central_lo,
            "hi": central_hi,
            "arm": {"kind": "central", "coeffs": coeffs},
            "eps": float(manifest["central_eps"]),
            "proof_kind": "gappa",
        },
        {
            "lo": central_hi,
            "hi": sat_hi,
            "arm": {"kind": "saturation", "value": 1.0},
            "eps": float(manifest["tail_pos_eps"]),
            "proof_kind": "gappa",
        },
    ]
    return {
        "boxes": boxes,
        "provenance": {
            "generator_version": "wi13-erf-envelope-optionb-2",
            # Arb cross-check parameters (the validate step re-derives the bound
            # at this precision/subdivision and asserts committed eps >= it).
            "certify_prec": 128,
            "certify_subdivisions": ARB_SUBDIVISIONS,
            "method": (
                "all arms Sollya + Gappa-proved (bundle sha256 "
                f"{manifest['bundle_sha256'][:16]}...); central = remez polynomial, "
                "tails = constant +-1, each proved by subdivision + certified "
                "Taylor model + Gappa; Arb whole-box certifier is the independent "
                "every-build cross-check"
            ),
            "gappa_bundle_sha256": manifest["bundle_sha256"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    if not PROOF_MANIFEST.exists():
        sys.stderr.write(
            f"error: Gappa proof manifest not found at {PROOF_MANIFEST}.\n"
            "Run: .venv/bin/python scripts/generate_erf_proof.py\n"
        )
        return 2
    manifest = json.loads(PROOF_MANIFEST.read_text())

    env = build_envelope(manifest)
    ENVELOPE_OUT.write_text(json.dumps(env, indent=2) + "\n")
    print(
        f"wrote {ENVELOPE_OUT.relative_to(REPO_ROOT)}: all three arms Gappa-proved "
        f"(central eps {float(manifest['central_eps']):.4e}, "
        f"tail eps {float(manifest['tail_pos_eps']):.4e})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
