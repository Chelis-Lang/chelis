# chelis#659 — Fuzz sampler transcendental cost blocks BS invariants

**Status:** Open (deferred pending concrete_eval fix)
**Severity:** Consumer-facing blocker
**Blocked by:** chelis#637 (relational/whole-expression abstraction), chelis#659 (this issue)
**Filed:** 2026-07-22
**Updated:** 2026-07-22

## Problem

The Tier C fuzz evaluator's `apply_intrinsic` whitelist did not include
`normal_cdf` or `erf`. Any property whose postcondition expression
involves `normal_cdf(x)` evaluated that call to `NaN`, causing:

- In postcondition position: every sample appeared to be a counterexample
  (since `NaN > 0.0` is false), producing a spurious disproof of the
  first sample
- The consumer-visible symptom: "one fuzz sample of one positivity
  property >200s" (actually a false-disproof, not a timeout per se, but
  the sampler configuration at the time looped on retries)

This single gap blocked the fuzz lane for every Black-Scholes invariant
that mentions `normal_cdf` in its expression.

## Gated Invariants

The following invariants are deferred in the consuming canon (`shoals`)
manifest and CANNOT re-enter the active canon until BOTH chelis#637 and
chelis#659 are resolved:

| Invariant | Property |
|-----------|----------|
| `call_price_nonneg` | `bs_call(s, k, r, σ, t) ≥ 0` for positive inputs |
| `call_monotone_in_s.bs` | `∂bs_call/∂s > 0` (delta positivity) |
| `bs_intrinsic_lower_bound` | `bs_call ≥ max(0, s − k·e^{−rt})` |
| `bs_vega_nonneg_grad` | `∂bs_call/∂σ ≥ 0` (vega non-negativity) |

These are listed in the manifest `deferred_invariants` at commit `e2cc9f1c`
and in `CHANGELOG.md` [0.22.0] Added section.

## Resolution Path

1. **chelis#659 (this issue):** Add `normal_cdf` and `erf` to
   `concrete_eval::apply_intrinsic`. This unblocks the fuzz lane: samples
   involving `normal_cdf` evaluate concretely rather than returning NaN.
   **Status: Fix implemented** (Abramowitz & Stegun 7.1.26 erf
   approximation, max error < 1.5e-7).

2. **chelis#637 (relational abstraction):** The SMT lane cannot prove BS
   positivity even with the envelope wired, because abstracting the
   coupled `normal_cdf(d1)` / `normal_cdf(d2)` is falsifiable in the
   over-approximation (the functional coupling `d2 = d1 − σ√t` is lost).
   This requires compound-argument interval propagation — the research-
   risk 60% identified in the probe_434 report.

## Cross-References

- `docs/diligence/notes/phase1/Chelis-Lang_shoals.json`: manifest evidence
- `CHANGELOG.md` [0.16.0]: confirms BS positivity stays `unsupported`
- `spec/design/probe_434_transcendental.md`: probe report (p17 confirms
  authoring conventions cannot sidestep compound propagation)
- `crates/chelis-prove/tests/transcendental_finance_lowering.rs`:
  regression lock for the honest `unsupported` verdict

## Acceptance Criteria

chelis#659 is CLOSED when:
- `chelis prove --tier fuzz` on the BS positivity property completes in
  < 5 seconds (was >200s)
- The fuzz lane for `normal_cdf`-involving properties evaluates concretely
  (no NaN rejection)
- A release binary carrying the fix is published

chelis#637 requires a separate scope decision (research-risk investment).
