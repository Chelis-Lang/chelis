# host-eval-perf-mc-rigor: 100K-path MC simulation does not terminate in the host evaluator

**Status:** open
**Filed:** 2026-05-01
**Owning phase:** chelis-core (host evaluator perf)
**Discovered by:** Phase 3l Shoals v0.1.0 manual rigor gate timing

## Summary

The Shoals manual rigor gate at `shoals/manual-gates/mc_rigorous.ch` runs
100K-path Monte Carlo pricing through the chelis host evaluator and does
not terminate in any reasonable wall-clock. Measured: >30 minutes on chelis
0.4.0 with no completion, on AMD Ryzen AI Max+ 395 hardware. The
manual-gate file shipped with Shoals v0.1.0 expecting "~10-15 minutes per
test" — that estimate was wrong; actual runtime is unbounded for practical
purposes.

The same Shoals MC code at the default-tier 20K paths runs in ~5 seconds
per test. The 5x path-count increase to 100K does not produce a 5x
wall-clock — interpreter overhead compounds nonlinearly, likely due to
allocation churn in the per-iteration sample/payoff loop.

This is not a regression. The original 10-15 min estimate was paper-design
without measurement. The architectural reality is: 100K-path numerical
loops are not practical on the host evaluator.

## Why this matters

Phase 3l's spec (`shoals/spec/phase3l.md` Test Plan, lines 71-72) requires:

> Monte Carlo price converges to Black-Scholes analytical for vanilla
> European call (< 1% error with 100K paths)

This is the load-bearing claim for the commercial pitch around verified
Monte Carlo pricing. Today the claim is design-correct (the algorithm is
right; 20K-path runs prove it converges), but the spec-rigor 100K-path
numerical assertion is not executable until the host evaluator gets faster
or Shoals can be AOT-compiled.

## Repro

```sh
cd /home/jeff/Documents/scratch/shoals
chelis test manual-gates/mc_rigorous.ch --timeout 3600
# Either of test_mc_call_rigorous, test_gbm_terminal_rigorous fails to
# complete within the 60-min ceiling on the spec'd hardware.
```

The default-tier comparison runs in seconds:

```sh
chelis test tests/  # 47 tests, ~3-5 min wall-clock total
```

## Required fix (one of the following)

1. **Host evaluator perf work.** Reduce per-iteration overhead (allocation,
   dispatch, closure-call costs) until 100K-path MC is sub-15-min on
   reference hardware. Likely candidates: cache compiled function bodies
   across iterations, reduce list allocations in `to_tensor`/`map`/`fold`
   chains, specialize numerical primitives.

2. **AOT path for shell packages.** Allow `chelis build --target c` (or
   equivalent) on a Shoals function so the rigor gate can run as compiled
   native code. The host evaluator stays usable for development; release
   gates compile to native.

(2) is more aligned with the trust-stack pitch since it gives customers a
build artifact to deploy. (1) is correct in any case for the inner-loop
ergonomics of `chelis test`.

Until either lands, the Shoals 100K-path rigor gate is documented as
deferred, not failing.
