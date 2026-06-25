#!/usr/bin/env python3
"""WI-13 Option B: assemble the committed erf envelope from the proof bundle.

Combines the two certified halves into the single committed envelope
``crates/chelis-prove/data/erf_envelope.json`` the runtime consumes:

  - CENTRAL arm: the Sollya remez polynomial and its Gappa-PROVED sup-norm eps,
    read from the Gappa proof manifest
    (``crates/chelis-prove/data/erf_proof/manifest.json``). proof_kind = gappa.
  - SATURATION tails: constant +-1 with an Arb-certified eps. proof_kind =
    arb_enclosure (Gappa has no erf/exp, so the erfc tail bound cannot be a
    Gappa proof term; the rigorous Arb enclosure is its certificate instead).

The per-box proof_kind makes the proof-vs-enclosure split machine-visible. The
tail eps is stamped by the Arb certifier; the whole envelope is then re-validated
by the Arb cross-check (``certify_erf_envelope validate``) so every committed
eps -- both Gappa and Arb boxes -- is independently confirmed to bound the truth
on every CI build.

Pipeline:
  1. .venv/bin/python scripts/generate_erf_proof.py        # central Gappa bundle
  2. .venv/bin/python scripts/assemble_erf_envelope.py     # this script
       (emits a draft; Arb-stamps tail eps; writes erf_envelope.json)
  3. cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
       -- validate crates/chelis-prove/data/erf_envelope.json   # cross-check
  4. git diff crates/chelis-prove/data/

Exit codes:
  0  envelope assembled and written
  2  the Gappa manifest is missing (run generate_erf_proof.py first)
  3  the Arb stamp step (cargo) failed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "crates" / "chelis-prove" / "data"
PROOF_MANIFEST = DATA_DIR / "erf_proof" / "manifest.json"
ENVELOPE_OUT = DATA_DIR / "erf_envelope.json"

# Saturation reach: the desk box hits the hundreds (master WI-13).
SAT_HI = 300.0
# Arb subdivision count, recorded in provenance and used by the Arb cross-check
# (`certify_erf_envelope validate`). It must be fine enough that the Arb sup-norm
# bound on the CENTRAL box drops below the Gappa-proved eps, so the cross-check
# (committed eps >= Arb bound) passes with the committed eps pinned to the
# Gappa-proved value. 2^21 puts the central Arb bound (~5.66e-7) just under the
# Gappa eps (~5.68e-7); the flat tails converge trivially at any count.
ARB_SUBDIVISIONS = 2097152


def build_draft(manifest: dict) -> dict:
    """Central arm from the Gappa manifest; tails as saturation placeholders
    (eps 0.0, stamped by Arb next)."""
    central_lo = float(manifest["central_lo"])
    central_hi = float(manifest["central_hi"])
    coeffs = [float(c) for c in manifest["coeffs"]]
    central_eps = float(manifest["central_eps"])

    boxes = [
        {
            "lo": -SAT_HI,
            "hi": central_lo,
            "arm": {"kind": "saturation", "value": -1.0},
            "eps": 0.0,
            "proof_kind": "arb_enclosure",
        },
        {
            "lo": central_lo,
            "hi": central_hi,
            "arm": {"kind": "central", "coeffs": coeffs},
            # Gappa-PROVED eps from the committed proof bundle.
            "eps": central_eps,
            "proof_kind": "gappa",
        },
        {
            "lo": central_hi,
            "hi": SAT_HI,
            "arm": {"kind": "saturation", "value": 1.0},
            "eps": 0.0,
            "proof_kind": "arb_enclosure",
        },
    ]
    return {
        "boxes": boxes,
        "provenance": {
            "generator_version": "wi13-erf-envelope-optionb-1",
            "certify_prec": 0,
            "certify_subdivisions": 0,
            "method": "DRAFT - central from Gappa proof bundle; tail eps stamped by Arb",
        },
    }


def arb_stamp_tails(draft_path: Path) -> dict:
    """Run the Arb certifier in `stamp` mode to fill the tail eps (it also
    re-certifies the central box; the central eps it writes will be the Arb
    bound, so we re-pin the Gappa-proved central eps afterward)."""
    env = dict(os.environ)
    env["CERTIFY_SUBDIVISIONS"] = str(ARB_SUBDIVISIONS)
    stamped = draft_path.with_suffix(".stamped.json")
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "chelis-prove",
            "--features",
            "arb",
            "--bin",
            "certify_erf_envelope",
            "--quiet",
            "--",
            "stamp",
            str(draft_path),
            str(stamped),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write("error: Arb stamp step failed.\n")
        raise SystemExit(3)
    result = json.loads(stamped.read_text())
    stamped.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    if not PROOF_MANIFEST.exists():
        sys.stderr.write(
            f"error: Gappa proof manifest not found at {PROOF_MANIFEST}.\n"
            "Run: .venv/bin/python scripts/generate_erf_proof.py\n"
        )
        return 2
    manifest = json.loads(PROOF_MANIFEST.read_text())

    draft = build_draft(manifest)
    draft_path = DATA_DIR / ".erf_envelope_draft.json"
    draft_path.write_text(json.dumps(draft, indent=2) + "\n")

    try:
        stamped = arb_stamp_tails(draft_path)
    finally:
        draft_path.unlink(missing_ok=True)

    # Pin the central arm's committed eps to exactly the Gappa-PROVED value, so
    # the committed band is precisely what the committed Gappa proof bundle
    # machine-checks -- the auditor's `gappa` run certifies the committed number.
    # proof_kind = gappa. The Arb cross-check (`validate`) independently confirms
    # this committed eps is >= the Arb sup-norm bound (ARB_SUBDIVISIONS is tuned
    # so the Arb bound lands just under the Gappa eps), i.e. Arb agrees the bound
    # is sound. The Arb stamp overwrote eps with the (slightly tighter) Arb bound;
    # we restore the Gappa value here.
    gappa_eps = float(manifest["central_eps"])
    # Restore the central coeffs directly from the manifest (not the Arb stamp's
    # serde round-trip, which re-serializes each f64 and can differ from the
    # manifest's parsed value by a ULP). The committed central polynomial must be
    # bit-identical to the one the Gappa proof is about, so it is sourced from
    # the manifest on both sides.
    manifest_coeffs = [float(c) for c in manifest["coeffs"]]
    for b in stamped["boxes"]:
        if b["arm"]["kind"] == "central":
            b["arm"]["coeffs"] = manifest_coeffs
            b["eps"] = gappa_eps
            b["proof_kind"] = "gappa"
        else:
            b["proof_kind"] = "arb_enclosure"

    # Record provenance tying the envelope to the Gappa proof bundle.
    stamped["provenance"]["method"] = (
        "central arm: Sollya remez + Gappa-proved sup-norm "
        f"(bundle sha256 {manifest['bundle_sha256'][:16]}...); "
        "tails: Arb-certified erfc enclosure"
    )
    stamped["provenance"]["gappa_bundle_sha256"] = manifest["bundle_sha256"]

    ENVELOPE_OUT.write_text(json.dumps(stamped, indent=2) + "\n")
    print(
        f"wrote {ENVELOPE_OUT.relative_to(REPO_ROOT)}: central eps {gappa_eps:.4e} (gappa), "
        f"tails Arb-stamped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
