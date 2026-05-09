# Canonical Reference Implementations for Domain Shells

## Purpose

Domain shells ship `references/` directories containing simple, obviously-correct reference implementations of standard domain models. These are the spec artifact for the spec-correspondence property category. Users verify their optimized implementations (or AI-generated implementations) against the references via `@property matches_reference forall(...)`.

The user does not write reference implementations for standard models. The shell provides them. The user writes references only for proprietary or custom models.

## Convention

Every domain shell that targets standard, well-defined models includes a `references/` directory alongside `properties/`:

```
chelis-lang/shoals/
├── src/                    # production implementations (optimized)
│   ├── black_scholes.ch
│   ├── heston.ch
│   └── ...
├── references/             # NEW: simple, obviously-correct implementations
│   ├── black_scholes.ch    # 5-15 lines per model, direct formula transcription
│   ├── heston.ch
│   ├── vasicek.ch
│   └── ...
├── properties/             # invariants and matches_reference checks
│   ├── pricing.ch
│   ├── greeks.ch
│   └── ...
└── tests/                  # deterministic tests via chelis test
```

## What Reference Implementations Look Like

A reference implementation is the textbook formula transcribed directly into Chelis. Optimized for readability and obvious-correctness, not performance. Vectorization, fusion, GPU dispatch are explicitly NOT done in references — those happen in `src/`.

Example: Black-Scholes call price reference (target: under 10 lines):

```chelis
-- references/black_scholes.ch
module Shoals.References.BlackScholes

import Std.Tensor (cast)
import Nautilus.Special (normal_cdf)

def call_price_reference(
  spot: f32, vol: f32, rate: f32, T: f32, strike: f32
) -> f32 = {
  d1 = div(
    add(log(div(spot, strike)), mul(add(rate, mul(0.5, mul(vol, vol))), T)),
    mul(vol, sqrt(T)))
  d2 = sub(d1, mul(vol, sqrt(T)))
  sub(
    mul(spot, normal_cdf(d1)),
    mul(mul(strike, exp(neg(mul(rate, T)))), normal_cdf(d2)))
}
```

The reference is the spec. A quant can verify it by inspection in 30 seconds. The optimized version in `src/` may be 50 or 200 lines (vectorized over many strikes, fused with Greeks computation, GPU-dispatched). The toolchain proves they agree on random inputs.

## What Reference Implementations Are NOT

- **Not "the reference is what runs in production."** References exist for verification. Production code is in `src/`.
- **Not feature-complete.** A reference covers the standard textbook case. Edge cases, extensions, variants are in `src/`.
- **Not a separate package.** References live inside the shell, not as a separate "shoals-references" reef package.
- **Not for proprietary models.** Domain shells provide references for standard models (Black-Scholes, Heston, Vasicek, etc.). Customers write their own references for proprietary models in their own packages.

## Required Reference Implementations Per Shell

### Shoals (finance)
- Black-Scholes call/put with continuous dividend yield
- Black-Scholes Greeks (delta, gamma, vega, theta, rho)
- Heston stochastic volatility (price only, basic version)
- Vasicek and CIR short-rate models
- Vanilla European Monte Carlo pricer
- Standard portfolio risk measures (VaR, CVaR via historical simulation)

### Octant (LaTeX bridge)
- Reference parsers for standard mathematical notation subsets
- Reference renderers for round-trip verification

### Future verticals
- Same pattern: domain shell ships canonical references for the standard models in that domain.

## Verification Pattern

The standard pattern in `properties/` files:

```chelis
-- properties/pricing.ch
import Shoals.BlackScholes (call_price)               -- production impl
import Shoals.References.BlackScholes (call_price_reference)

@property matches_textbook_reference forall(
  spot: f32, vol: f32, rate: f32, t: f32, strike: f32
) where spot > 0.0, vol > 0.0, t > 0.0, strike > 0.0:
  close(
    call_price(spot, vol, rate, t, strike),
    call_price_reference(spot, vol, rate, t, strike),
    1e-6)
```

`chelis prove src/black_scholes.ch --samples 1000` runs this property against deterministic random inputs from the binder types and verifies agreement.

## Customer Workflow

1. Customer installs Shoals.
2. Customer runs `chelis prove src/` on their own pricing code that imports Shoals primitives.
3. Shoals' canonical properties verify standard invariants (put-call parity, delta bounds, etc.) and reference correspondence (matches_textbook_reference) against Shoals' production implementations.
4. Customer writes their own properties for proprietary aspects of their models.
5. Customer writes their own references only for proprietary models that don't have a textbook formula.
6. CI gate: `chelis prove` runs on every commit. Property failures block deployment.

The customer's investment in writing references scales with how proprietary their models are. For a shop using mostly standard models with custom calibration, references come from Shoals and the customer writes only properties (which are short and declarative).
