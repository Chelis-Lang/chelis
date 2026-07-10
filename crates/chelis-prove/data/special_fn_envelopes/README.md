# Special-function envelope generation configs (chelis#434)

This directory holds the **generation configs** for the `exp`/`log`/`sqrt`
certified envelopes the abstract-subterm finder
(`src/transformations/abstract_subterm.rs`) will consume once their data lands.
It does **not** yet hold committed envelope JSON: producing a *certified* envelope
needs a sound-eps anchor, and neither anchor is exercisable today (see the gap
below). Until then, `SpecialFnEnvelope::committed(f)` returns `None` for these
functions and the finder **declines** — the honest floor.

The certified `erf` envelope (`../erf_envelope.json` + `../erf_proof/`) is the
reference implementation of the full recipe.

## The recipe (from `docs/erf_envelope_regen.md`)

1. **Float proposer** (`scripts/generate_erf_envelope.py`): `numpy.polyfit` at
   Chebyshev nodes against an mpmath high-precision truth → central polynomial
   coefficients per box. Function-agnostic: swap `mpmath.erf` for the target.
   **Verified to generalize** — probe p19 fit `exp` over `[-2,2]` to a 1.76e-12
   empirical sup-norm error (`spec/design/probe_434_transcendental.md`).
2. **Sound-eps anchor** (either is sufficient; `erf` carries both):
   - *Sollya → Gappa proof bundle* (`scripts/generate_erf_proof.py`): Sollya's
     remez + certified per-sub-interval Taylor model, Gappa machine-checks
     `|p − T| ≤ bound`. Produces `data/<fn>_proof/*.gappa`.
   - *Arb whole-box certifier* (`src/arb_oracle.rs`, `--features arb`): rigorous
     ball-arithmetic sup-norm enclosure stamps each box's `eps`.

## The gap (why no committed data yet)

Both anchors are erf-specialized or environment-blocked as of this slice:

- **Sollya not installed** here (`.local/bin/sollya` absent;
  `scripts/setup_proof_toolchain.py` builds it from source). `gappa` itself IS
  present — only the Sollya remez/Taylor-model half is the env gap.
- **Arb certifier hardcodes `arb_hypgeom_erf`** (`arb_oracle.rs::rigorous_erf_enclosure`).
  The `ErfEnclosure` framework carries no special-function identity, so
  generalizing is a swap to `arb_exp` / `arb_log` / `arb_sqrt` plus function-
  keying — mechanical, but a code change, and the `arb` feature vendors FLINT/Arb
  (heavy build). This is part of the deferred de-narrowing, not this probe slice.

When either anchor is unblocked, `<fn>.config.json` drives the proposer and the
certifier to produce `data/<fn>_envelope.json`, which `SpecialFnEnvelope::committed`
then serves.

## Config schema

Each `<fn>.config.json`:

- `function` — the special-function name (must be in `SpecialFnRegistry::known_functions()`).
- `domain` — `all_reals` | `positive` | `non_negative` (must match `SpecialFnRegistry::domain`).
- `output_clamp` — `[lo, hi]` global output range, or `null`.
- `boxes` — the intended piecewise decomposition: each `{lo, hi, arm, degree}`
  (`arm` = `central` | `saturation`). Finance-realistic argument ranges.
- `certify_status` — `"BLOCKED: <reason>"` until the anchor is unblocked.

The boxes/degrees are proposals sized to the finance argument ranges; the
certifier's `eps` is what makes them sound, and a poor fit only enlarges `eps`
(still sound), never breaks soundness.
