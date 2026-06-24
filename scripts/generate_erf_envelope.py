#!/usr/bin/env python3
"""Propose the central polynomial coefficients for the WI-13 erf envelope.

This is the FLOAT PROPOSER half of the envelope generator. It emits a
coefficients-only DRAFT (boxes + saturation arms + central polynomial
coefficients, but NO certified `eps`). The Arb oracle is the trust
anchor that stamps `eps`: run the Rust certifier
(`cargo run -p chelis-prove --features arb --bin certify_erf_envelope`)
on this draft to produce the committed
`crates/chelis-prove/data/erf_envelope.json`.

The split is deliberate. A wrong proposed polynomial here cannot make
the envelope unsound: the Arb certifier computes `eps` as a rigorous
sup-norm bound of `|p(x) - erf(x)|` over each whole box, so a poor fit
only yields a LARGER (still sound) `eps`. Soundness rests entirely on
the Arb certification step, never on this float computation. This
mirrors the Clarabel SoS engine's float-proposer / exact-verifier split.

Pipeline:
  1. python3 scripts/generate_erf_envelope.py > /tmp/erf_envelope_draft.json
  2. cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
       -- /tmp/erf_envelope_draft.json \
       crates/chelis-prove/data/erf_envelope.json
  3. git diff crates/chelis-prove/data/erf_envelope.json
  4. (CI re-validates the committed eps against Arb on every build)

Envelope shape (master WI-13 + the bake-off):
  - erf is odd, so we cover the whole range and rely on the central
    polynomial being a good odd-ish fit; the saturation arms are exact
    constants +-1 (erf is within ~1e-17 of +-1 outside +-6).
  - boxes (contiguous, ascending):
      [-SAT_HI, -CENTRAL] saturation -1
      [-CENTRAL, +CENTRAL] central polynomial
      [+CENTRAL, +SAT_HI]  saturation +1
    where CENTRAL = 6.0 (erf saturates there) and SAT_HI = 300.0
    (the desk box reaches the hundreds).

Dependencies (regen env, for the dep-setup docs):
  - the uv-managed Python at .venv/bin/python (>=3.11)
  - numpy (least-squares solve of the Remez-style fit)
  - mpmath (arbitrary-precision erf truth for the fit targets)
  The CERTIFY step additionally needs the Rust `arb` feature build
  deps (gcc, m4, make; CFLAGS=-fPIC; the *_CACHE dirs); see
  crates/chelis-prove/src/arb_oracle.rs and .cargo/config.toml.

Exit codes:
  0  draft written to stdout
  2  a required dependency (numpy/mpmath) is missing
"""

from __future__ import annotations

import argparse
import json
import sys

GENERATOR_VERSION = "wi13-erf-envelope-1"

# Central region half-width. erf is within ~1e-17 of +-1 beyond +-6, but it is
# already within ~2.2e-5 of +-1 by +-3, so the central polynomial only needs to
# cover [-CENTRAL, CENTRAL] = [-3, 3] and the saturation arm absorbs [3, 300]
# with an eps of ~2.2e-5 (the value of 1 - erf(3)). A wider central box forces
# huge monomial coefficients (x^deg over [-6,6] reaches ~1e11), which both fits
# worse in the ill-conditioned monomial basis AND blows up the ball-arithmetic
# certification; +-3 keeps the monomial values tame so the Arb certifier returns
# a tight, true sup-norm eps (~1e-4 at degree 15) rather than a dependency-
# problem-dominated bound.
CENTRAL = 3.0
# Saturation reach: the finance desk box hits the hundreds (master WI-13).
SAT_HI = 300.0
# Central polynomial degree. erf is odd; a moderate odd-capable degree fits the
# central region well. The Arb certifier stamps the true sup-norm error, so this
# is a fit-quality knob, not a soundness knob. Degree 21 over [-3, 3] certifies
# to a sup-norm error of ~6.5e-7, a tight, useful envelope; higher degrees gain
# little against the monomial-basis conditioning floor.
DEGREE = 21
# Number of least-squares fit nodes across the central box (Chebyshev nodes
# concentrate sampling near the edges where the fit is hardest).
FIT_NODES = 400


def _require(mod_name: str):
    try:
        return __import__(mod_name)
    except ImportError:
        sys.stderr.write(
            f"error: missing dependency '{mod_name}'. Install the regen env:\n"
            f"  uv pip install numpy mpmath\n"
        )
        raise SystemExit(2)


def central_coeffs(degree: int, n_nodes: int) -> list[float]:
    """Least-squares fit of erf on [-CENTRAL, CENTRAL] at Chebyshev nodes,
    using mpmath erf as the high-precision target. Returns coefficients in
    DESCENDING degree order (numpy polyfit convention), matching the Rust
    `ErfArm::Central` Horner evaluation.
    """
    numpy = _require("numpy")
    mpmath = _require("mpmath")
    np = numpy
    mpmath.mp.dps = 50

    # Chebyshev nodes mapped to [-CENTRAL, CENTRAL]: denser near the edges.
    k = np.arange(n_nodes)
    cheb = np.cos((2 * k + 1) * np.pi / (2 * n_nodes))
    xs = CENTRAL * cheb
    ys = np.array([float(mpmath.erf(mpmath.mpf(float(x)))) for x in xs])

    # Plain least-squares polynomial fit. (Remez would equioscillate the error;
    # least-squares is close enough as a PROPOSER since Arb stamps the true
    # error. Descending-degree coeffs.)
    coeffs = np.polyfit(xs, ys, degree)
    return [float(c) for c in coeffs]


def build_draft(degree: int = DEGREE, n_nodes: int = FIT_NODES) -> dict:
    coeffs = central_coeffs(degree, n_nodes)
    # Boxes are contiguous and ascending. eps is OMITTED here (null) and stamped
    # by the Arb certifier.
    boxes = [
        {"lo": -SAT_HI, "hi": -CENTRAL, "arm": {"kind": "saturation", "value": -1.0}, "eps": 0.0},
        {"lo": -CENTRAL, "hi": CENTRAL, "arm": {"kind": "central", "coeffs": coeffs}, "eps": 0.0},
        {"lo": CENTRAL, "hi": SAT_HI, "arm": {"kind": "saturation", "value": 1.0}, "eps": 0.0},
    ]
    return {
        "boxes": boxes,
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            # certify_prec / certify_subdivisions / method are filled by the
            # Rust certifier; placeholders here so the draft round-trips.
            "certify_prec": 0,
            "certify_subdivisions": 0,
            "method": "DRAFT (uncertified) - run certify_erf_envelope",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degree", type=int, default=DEGREE, help="central polynomial degree"
    )
    parser.add_argument(
        "--nodes", type=int, default=FIT_NODES, help="number of fit nodes"
    )
    args = parser.parse_args(argv)

    draft = build_draft(args.degree, args.nodes)
    json.dump(draft, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
