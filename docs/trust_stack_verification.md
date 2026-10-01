# Trust-Stack Verification Architecture

This document describes the three-tier proof pipeline behind `chelis prove`
and where it sits in the Chelis trust stack.

## Layers

```
┌─────────────────────────────────────────────────────────┐
│  Specification Consumption (chelis-prove)                │
│  @property → Tier A/B/C dispatch → proof artifact       │
├─────────────────────────────────────────────────────────┤
│  Per-Program Runtime Verification (chelis prove fuzz)   │
│  @property → randomized testing → statistical confidence│
└─────────────────────────────────────────────────────────┘
```

Requirements enter this stack as Chelis Surf `@property` declarations with
parameters, preconditions, and postconditions. A tool that translates
requirements written in another notation emits those declarations; the proof
pipeline does not depend on where they came from.

**chelis-prove** is the specification-consumption layer. It takes Chelis
properties and dispatches them through three verification tiers:

- **Tier A (Type System):** Structural validation via the existing Chelis type
  checker. Catches ill-formed properties. Positively discharges dimension-type,
  effect-row, and linearity properties without runtime cost.

- **Tier B (SMT):** Lowers predicates to SMT terms and solves them with cvc5
  (the `smt` feature), with Z3 as an optional second engine (the `z3` feature)
  for goals cvc5 leaves unknown. Proves or disproves properties over nonlinear
  real arithmetic, polynomial bounds, and, where the solver handles them,
  transcendental expressions.

- **Tier C (Fuzz):** Randomized property-based testing with seeded PRNG.
  Statistical validation when Tier A is inconclusive and Tier B times out or is
  not amenable.

- **COMPOSE:** Folds each proof with the producer/contract assumptions it
  depends on. Result JSON carries the weakest-badge `composite_verdict` token,
  a `qualifiers` array with the full disclosed caveat set, and per-assumption
  `discharge:{method,evidence}` records. A fuzz-only BASE (no proof underneath)
  composes to `fuzz_validated` and never to a `proven_*` badge; a
  sound-over-approximation base composes to `sound_approximate`. For an
  SMT-proved base: all-SMT discharges compose to `proven_modulo_real_arithmetic`
  (the proof is over the reals, not machine arithmetic); a fuzz CONTRACT
  discharge is qualified `proven_modulo_fuzz_validated_contract`; asserted
  axioms are qualified `proven_modulo_asserted_axiom`. Plain `proven` is
  reserved for a future exact-machine-arithmetic lowering.

**chelis prove** is also the per-program runtime verification layer. It
discovers `@property` declarations and evaluates them with random inputs. The
three-tier dispatcher extends this with static verification for properties that
admit it.

## Design Constraints

- **Predicate lowering**: how Chelis predicates map to SMT terms. Tier B
  inlines producer bodies before lowering.
- **Soundness preservation**: a timeout or unknown answer is never reported as a
  proof, or as a failure without a counterexample.
- **Solver abstraction**: each solver is a discharge engine behind a shared
  interface, so engines can be added or swapped without architectural change.

## Relationship to the Type System

The Chelis type system (dimensions, effects, linearity) provides Tier A's
positive-discharge capability. Properties that the type system can prove are
certified at compile time with no runtime cost, which is strictly stronger than
SMT proof for the cases it handles.

Compiler correctness is a separate layer: differential testing of the compiler
against a reference checker and evaluator asks whether the compiler implements
the language specification. The tiers above ask whether a user's program
satisfies the user's `@property` requirements. The two layers are complementary,
not duplicates.

## Scope Limits

- Tier B timeout/unknown is unsupported with a reason on smt-only paths, or
  Tier C fallback in auto mode. It is never reported as failed without a
  counterexample.
- Tier B emits cvc5's partial `sqrt` only when a separate total-algebraic
  obligation proves every argument non-negative from the user's independent
  conjunctive preconditions. Unproved, nested, quantified, or partial-evidence
  shapes fall through to Tier C rather than strengthening the query. The
  auxiliary and main solver phases share the request timeout; their aggregate
  solver budget does not exceed it, while non-`sqrt` requests retain the whole
  timeout.
- Assumption-backed green proofs run a non-vacuity check over the assumptions
  alone. SAT establishes the assumption domain, UNSAT makes the composed
  verdict `invalid`, and unknown/timeout makes it `unsupported`.
