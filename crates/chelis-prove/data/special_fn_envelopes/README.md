# Special-function envelopes: certified data + generation (chelis#434)

Function-keyed certified envelopes for `{exp, log, sqrt}` — the generalization of
the certified `erf` envelope (`../erf_envelope.json` + `../erf_proof/`, the
reference implementation of the full recipe) that the abstract-subterm finder
(`src/transformations/abstract_subterm.rs`) consumes.

## Status

| fn    | committed data              | anchor        | notes |
|-------|-----------------------------|---------------|-------|
| `erf` | `../erf_envelope.json`      | Gappa + Arb   | `[-300,300]`, degree-21 central + saturation tails |
| `exp` | `exp_envelope.json`         | Arb mean-value| `[-2,2]`, degree-12, `eps ~1.4e-8` |
| `log` | `log_envelope.json`         | Arb mean-value| `[0.3,3.5]`, degree-12, `eps ~1.5e-4` |
| `sqrt`| `sqrt_envelope.json`        | Arb mean-value| `[0.04,4]`, degree-12, `eps ~2.4e-3` |

`SpecialFnEnvelope::committed(f)` serves all four; the finder abstracts a
transcendental site only when the argument's interval is inside the covered box
(else it DECLINES — the honest floor). `exp`/`log`/`sqrt` use the **mean-value**
Arb certifier (tight), cross-checked `<= naive` and `validate`-passing; each box
is bounded away from the derivative singularity at 0 (`log`/`sqrt` need `lo>0`).

## The recipe (generalizes `docs/erf_envelope_regen.md`)

1. **Float proposer** — `scripts/generate_special_fn_envelope.py {exp|log|sqrt}`:
   `numpy.polyfit` at Chebyshev nodes against an mpmath truth → central coeffs
   per box (emitted as exact hex-float strings). Reads `<fn>.config.json`.
2. **Arb certifier** — `cargo run -p chelis-prove --features arb --bin
   certify_special_fn_envelope -- stamp <draft.json> <out.json>`: stamps each
   box's rigorous sup-norm `eps` via `certify_sup_norm_over_box_general`
   (function-general naive whole-box ball arithmetic), `proof_kind = arb_enclosure`,
   with a 1-ppb safety margin, and records the certify subdivisions in
   `provenance`. Soundness rests entirely on this step; a poor fit only enlarges
   `eps`.
3. **Cross-check (CI / manual gate)** — `... -- validate <envelope.json>`:
   re-certifies at the recorded subdivisions and asserts every committed `eps`
   `>=` the fresh Arb bound. Run this whenever the data or the certifier changes.

The `exp` data above was produced by exactly this pipeline and passes `validate`.

## The gap (why exp is `[-2,2]`, and why log/sqrt are still BLOCKED)

The **naive whole-box** Arb certifier is SOUND for any subdivision but LOOSE (the
ball-Horner dependency problem): over `[-2,2]` it stamps `eps ~2.8e-5` where the
true fit error is `~4e-10`. It gets much looser on:

- **wider / higher-degree boxes** (e.g. `exp` over `[-15,2]` — the full finance
  range), and
- **boxes near a domain edge** — `log` near `0` (log → −∞, huge variation) and
  `sqrt` near `0` (derivative → ∞). Their finance boxes reach the edge, so the
  naive bound is near-vacuous there.

A TIGHT `eps` on those needs the **mean-value form** the erf certifier uses
(`certify_sup_norm_over_box`), which requires each function's closed-form
derivative enclosure — clean for `exp` (`exp' = exp`), but `log' = 1/x` and
`sqrt' = 1/(2√x)` need near-0 care. Generalizing the mean-value certifier (and/or
writing per-function Sollya→Gappa drivers — Sollya IS installed and re-validates
the erf bundle here) is the scoped follow-up that unblocks tight `exp` over the
full range and `log`/`sqrt`.

## Config schema (`<fn>.config.json`)

- `function` — must be in `SpecialFnRegistry::known_functions()`.
- `domain` — `all_reals | positive | non_negative` (must match `SpecialFnRegistry::domain`).
- `output_clamp` — `[lo, hi]` or `null`.
- `boxes` — `{lo, hi, arm, degree}` decomposition.
- `certify_status` — `"CERTIFIED: …"` (committed) or `"BLOCKED: …"` (data pending).
  A test (`special_fn_envelope::tests::generation_configs_are_consistent_with_the_registry`)
  locks `committed()`-presence to this field.
