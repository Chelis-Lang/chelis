# Trust-Stack Verification Architecture

This document frames the three-tier proof pipeline as architecture work within
the Chelis trust stack.

## Three Layers

```
┌─────────────────────────────────────────────────────────┐
│  Specification Translation (c-earchin)                  │
│  EARS requirements → Chelis @property declarations      │
├─────────────────────────────────────────────────────────┤
│  Specification Consumption (chelis-prove)                │
│  @property → Tier A/B/C dispatch → proof artifact       │
├─────────────────────────────────────────────────────────┤
│  Per-Program Runtime Verification (chelis prove fuzz)   │
│  @property → randomized testing → statistical confidence│
└─────────────────────────────────────────────────────────┘
```

**c-earchin** is the specification-translation layer. It parses EARS
requirements (optionally with function bindings and typed predicates),
classifies them, and emits Chelis Surf `@property` declarations with real
parameters, preconditions, and postconditions.

**chelis-prove** is the specification-consumption layer. It takes Chelis
properties and dispatches them through three verification tiers:

- **Tier A (Type System):** Structural validation via the existing Chelis type
  checker. Catches ill-formed properties. Positively discharges dimension-type,
  effect-row, and linearity properties without runtime cost.

- **Tier B (SMT — cvc5):** Lowers predicates to SMT terms and solves with cvc5.
  Proves or disproves properties over nonlinear real arithmetic, polynomial
  bounds, and (where cvc5's CAD handles them) transcendental expressions.

- **Tier C (Fuzz — existing):** Randomized property-based testing with seeded
  PRNG. Statistical validation when Tier A is inconclusive and Tier B times out
  or is not amenable.

**chelis prove (existing)** is the per-program runtime verification layer. It
discovers `@property` declarations and evaluates them with random inputs. The
three-tier dispatcher extends this with static verification for properties that
admit it.

## Why This Is Architecture Work

The cvc5 integration is structurally distinct from anything else in the chelis
workspace. It is the first SMT-backed static verification layer. The decisions
made here constrain everything that follows:

- **Solver choice** (cvc5 over Z3): better NRA/CAD for finance queries.
- **Predicate lowering**: how Chelis predicates map to SMT terms.
- **Soundness preservation**: v0.1 only inlines function bodies expressible in
  SMT (polynomial, conditional). No uninterpreted function abstraction (risk of
  incomplete axioms producing false proofs).
- **Trait abstraction**: `Solver` trait allows future solver swaps without
  architectural change.

## Relationship to Hull

Hull proves the `@property + matches_reference` pattern at compiler scale
(numerical parity between Chelis IR evaluator and C backend). This workstream
extends `@property` from runtime sampling to SMT-discharged static verification
for predicates that admit it. They are complementary trust-stack expansions.

## Relationship to LaCaDiLE

The LaCaDiLE type system (dimensions, effects, linearity) provides Tier A's
positive-discharge capability. Properties that the type system can prove are
formally certified at compile time with no runtime cost — strictly stronger than
SMT proof for the cases it handles.

## For the C Proof Quant-Finance Pitch

This closes the loop from "we test our compiler with this pattern" to "we
statically verify your XVA invariants with this pattern." Hull provides
credibility (compiler correctness). This workstream provides user-facing
capability (requirement verification). Together they form the trust stack that
DORA, SR 11-7, and EU AI Act compliance buyers need.

## Scope Limits (v0.1)

- Only inlineable functions (polynomial arithmetic, conditionals, no recursion).
- Non-inlineable functions force Tier C (fuzz fallback).
- No uninterpreted function abstraction until v0.2 (soundness risk).
- Tier A does NOT discharge range/bounds/monotonicity (requires refinement types
  not yet in chelis-types).
- Tier B timeout → Inconclusive → Tier C. Fully automatic, no human review.

## Multi-Domain Evidence (v0.1)

The three-tier architecture is demonstrated across six domains:

- **Finance** (DORA, SR 11-7): option pricing, put-call parity, Black-Scholes
- **Healthcare** (FDA, IEC 62304): drug dosing, dose ceilings, pacemaker bounds
- **Aerospace** (DO-178C, FAA): control surface limits, thrust conservation, orbital energy
- **Automotive** (ISO 26262): ABS braking, battery thermal, steering torque
- **Manufacturing** (ISO 9001, cGMP): reactor setpoints, mass balance, pressure vessels
- **Power Systems** (NERC, IEC 61850): bus balance, frequency deadband, inverter ride-through

28 properties total. 22 proved by SMT (Tier B). 6 validated by fuzz (Tier C).
0 failed. The same verification machinery handles all domains — the `@property`
is the specification; the three-tier dispatcher discharges what it can; the rest
gets statistical validation. The audit trail is mechanical.

This is evidence that the architecture is general-purpose, not domain-bespoke.
The OOPSLA contribution claim is a generalization: "the same architecture
verifies properties across six domains with different compliance regimes."
