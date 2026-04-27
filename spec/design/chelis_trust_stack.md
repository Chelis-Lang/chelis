# Chelis Trust Stack for Consequential Computing

**Purpose:** Define the layered verification capabilities Chelis offers to users deploying generated code in high-stakes domains (finance, defense, aerospace, healthcare, scientific computing). This document captures the complete vision: what's shipped, what's planned, and what's future.

---

## The Problem

AI coding agents generate code that compiles, passes tests, and silently computes wrong answers. OpenAI's analysis of SWE-bench Verified found that 60% of remaining "unsolved" problems had flawed test cases that rejected correct solutions. All frontier models had memorized benchmark solutions from training data. Test suites are unreliable as the sole verification mechanism.

For consequential computing -- domains where millions of dollars or human life depend on code correctness -- the question is not "does the code compile?" or "does it pass tests?" but "does the code match the user's intent, and can we prove it?"

The gap between intent and implementation is where cost, delay, and risk concentrate. Manual code review is the current bridge, but AI-generated code makes the gap wider (more code, faster) and the review harder (the reviewer didn't write it).

## The Trust Stack

Three layers, ordered by what's shipped first and what the user interacts with.

### Level 1 -- Compiler Guarantees (shipped, free, automatic)

The Chelis type system statically checks properties that matter for numerical computing. These are structural soundness properties: they eliminate entire bug classes without user effort.

| Property | What it catches | How |
|---|---|---|
| Dimension safety | Shape mismatches, transposed matrices, broadcasting errors | Named dimensions checked at compile time |
| Effect tracking | Untracked randomness, missing I/O declarations, unhandled GPU resources | Effects in function signatures, propagated by the type checker |
| Linearity | Use-after-free on GPU tensors, memory leaks in long-running processes | Each tensor consumed exactly once |
| Differentiability | Wrong gradients from non-differentiable operations, in-place mutation inside grad | `grad(f)` only compiles if f is pure and differentiable |
| Reproducibility | Unseeded Monte Carlo, non-deterministic simulation | `chelis manifest --check` verifies every Random operation is seeded |

These guarantee that a program is internally consistent. They do NOT guarantee it computes the right answer. A program can be dimension-safe, effect-correct, linear, and differentiable while implementing the wrong formula entirely.

### Level 2 -- Executable Properties as Spec (planned, core future value prop)

The user writes executable Chelis functions that define what "correct" means in their domain. These properties ARE the spec. The toolchain verifies the generated code against them empirically on random inputs.

This is the capability that addresses the customer's concern directly: "I can't define a spec and ensure it's actually in the generated code." The answer: "You define the spec as executable properties. We verify the code satisfies them."

**Properties are Chelis functions that return bool:**

```chelis
-- Domain invariants: things that must always be true
@property fn price_is_positive(spot, vol, rate, T, strike) -> bool =
  gt(price(spot, vol, rate, T, strike), 0.0)

@property fn delta_in_unit_interval(spot, vol, rate, T, strike) -> bool =
  let d = grad(price, wrt=spot)(spot, vol, rate, T, strike)
  in and(gte(d, 0.0), lte(d, 1.0))

@property fn put_call_parity_holds(spot, vol, rate, T, strike) -> bool =
  close(
    sub(call_price(spot, vol, rate, T, strike),
        put_price(spot, vol, rate, T, strike)),
    sub(spot, mul(strike, exp(neg(mul(rate, T))))),
    1e-4)

-- Spec correspondence: the complex code matches the simple formula
@property fn matches_textbook(spot, vol, rate, T, strike) -> bool =
  close(
    price(spot, vol, rate, T, strike),
    textbook_black_scholes(spot, vol, rate, T, strike),
    1e-6)
```

The last pattern -- `matches_textbook` -- is the highest-value property type. The user (or their quant) writes a 5-line direct transcription of the formula from a textbook. It's obviously correct by inspection. The AI generates the optimized `price` function (vectorized, fused, GPU-dispatched). The toolchain runs both on 100,000 random inputs and verifies they agree.

**The customer doesn't review the generated code. They review the properties.** Properties say what correct means. If the properties are right and the code satisfies them, the code is right. If the AI generates code that violates a property, it doesn't ship.

**Three property categories:**

1. **Domain invariants.** Things that must always be true regardless of implementation: output is non-negative, delta is between 0 and 1, put-call parity holds, portfolio weights sum to 1. These are cheap to state and catch a wide class of implementation bugs.

2. **Spec correspondence.** The optimized implementation matches a simple reference implementation on random inputs. The reference implementation IS the spec -- it's a direct, obviously-correct transcription of the formula. The optimized implementation is what runs in production. Disagreement on any input is a definitive bug.

3. **Behavioral constraints.** Monotonicity (price increases with spot), continuity (small input change produces small output change), symmetry (put-call symmetry), convergence (Monte Carlo converges as path count increases). These encode domain knowledge that test suites can't express.

**Implementation: evolution of `chelis fuzz`**

The currently planned `chelis fuzz` accepts CLI flags for simple properties. The evolution:

- Properties are Chelis functions (not CLI flags). They live in `properties/*.ch` alongside the code, version-controlled, CI-enforced.
- `chelis fuzz src/pricer.ch` discovers `@property` annotations and tests each on random inputs drawn from the parameter types.
- Random input generation is type-directed: `f32` draws from a configurable range, `tensor[n, f32]` draws element-wise, `bool` draws uniformly.
- Failure on any input produces the failing input as a minimal reproducible case: "property `price_is_positive` violated at spot=0.001, vol=3.5, rate=-0.02, T=0.001, strike=1000.0".
- `chelis fuzz --trials 100000` controls the number of random inputs. Default: 10,000.
- CI integration: `chelis fuzz` runs as a CI gate alongside `chelis test`. Properties must hold on all sampled inputs for the build to pass.

**Differential testing is a special case of property-based fuzzing:**

```chelis
@property fn matches_reference(x: tensor[n, f32]) -> bool =
  close(optimized_fn(x), reference_fn(x), 1e-6)
```

No separate differential testing infrastructure needed. It's a property that compares two implementations. The fuzzer handles the random input generation and the agreement checking.

**Canonical domain properties ship with domain shells.** Every domain shell includes a `properties/` directory containing reference `@property` functions for the domain's standard invariants. These are not tests (they live alongside `tests/`, not inside it). They are verification contracts that demonstrate the `@property` pattern on real domain code.

For Shoals (finance):

- `properties/pricing.ch` -- put-call parity, price positivity, call bounded by spot, delta in [0,1], gamma positive for vanilla Europeans
- `properties/greeks.ch` -- grad-derived Greeks match finite-difference Greeks within tolerance, vega positive for vanilla options
- `properties/monte_carlo.ch` -- Monte Carlo price converges to analytic price as path count increases, variance decreases with path count
- `properties/no_arbitrage.ch` -- no-arbitrage conditions on option spreads (bull spread payoff non-negative, butterfly spread payoff non-negative)

For Octant (LaTeX bridge):

- `properties/roundtrip.ch` -- `parse(render(expr))` recovers the original expression; `lower(parse(latex))` type-checks
- `properties/provenance.ch` -- every Deep AST node has a provenance span that points inside the original LaTeX string

For future vertical shells: the same pattern. Canonical properties are the first thing a new domain shell ships, alongside the implementation code they verify.

The onboarding story: "Install Shoals. Run `chelis fuzz src/`. See 15 canonical finance properties pass on the reference implementations. Now write your own pricing model and add your own properties." The properties are documentation-by-example, not a separate product.

The convention is a hard rule: properties are co-located with the implementation they verify, NOT packaged as a separate "properties" shell. The `@property` infrastructure is in the compiler (`chelis fuzz`); the properties themselves are domain-specific and belong inside the domain shell. A standalone "finance properties" package with no implementation code is an empty vessel.

### Reference Implementations as Spec Artifacts

The "spec correspondence" property category requires a reference implementation to compare against. For standard domain models, the user does not write the reference — the domain shell provides it. Domain shells ship a `references/` directory alongside `properties/`, containing simple, obviously-correct implementations of standard models (Black-Scholes, Heston, Vasicek for finance; standard control laws for aerospace; etc.).

The customer writes:
- Properties (declarative invariants and reference-correspondence checks)
- References for proprietary models that don't have a textbook formula

The customer does NOT write:
- References for standard models (provided by the shell)
- The optimized implementations themselves (these are AI-generated or human-written, verified against references)

This addresses the failure mode where AI generates code that's mathematically valid but implements the wrong of two plausible relationships. The reference is the disambiguator. The user reads the reference (5-10 lines, obvious-by-inspection) instead of the optimized implementation (50-200 lines, opaque).

Full design: `chelis_reference_implementations_spec.md`.

### Level 3 -- Automated Static Analysis (future, power-tool, opt-in)

A specialized analysis tool (working name: Beacon) that runs on the compiled tensor DAG and computes over-approximations of value ranges at each node. The user doesn't annotate anything. The tool infers properties automatically.

```bash
chelis analyze --input-ranges "spot:[0.01,10000] vol:[0.001,5.0] rate:[-0.1,0.5]" src/pricer.ch
```

Output:
```
pricer.ch:12  div(x, vol)     -- vol may be zero at vol=0.001 lower bound.
                                 Add guard or tighten input range.
pricer.ch:18  exp(rate * T)   -- rate * T in [-50.0, 250.0]. exp(250) overflows f32.
                                 Use f64 or clamp.
pricer.ch:25  result          -- bounded in [0.0, 9987.3] for specified input ranges.
                                 No NaN on any path.
```

Computationally expensive (minutes, not milliseconds). A pre-deployment gate, not an inner-loop tool. Inspired by Astree, which verified the absence of runtime errors in the Airbus A380 flight control software.

Not currently designed. Recorded as a future shell (Beacon) in the ecosystem. Depends on the tensor DAG being stable (it is) and the input range specification mechanism (needs design).

**Contract evolution:** Phase 5g's trusted annotations (`@convex`, `@lipschitz`) start as documentation. As Beacon matures, it can verify some of these annotations automatically: check whether `@convex` actually holds by analyzing the second derivative's sign over the specified input ranges. The annotations don't change; the verification level increases over time.

### What's NOT in the stack

**Refinement types (SMT-verified predicates on types).** Dropped. They add programming model complexity (users must understand Z3, restructure code to help the solver, debug incomprehensible solver failures) and don't address the actual concern (does the code match intent). Properties expressed as executable Chelis functions and tested empirically are more actionable, more debuggable, and more accessible than SMT-verified type predicates.

**Runtime monitoring as a language feature.** Dropped. If constraints are captured at compile time (Level 1) and empirically verified pre-deployment (Level 2), runtime monitoring inside the program is redundant. Boundary validation (API inputs, database results, external data feeds) is an application concern handled by library code at I/O boundaries, not a language feature.

**Full formal verification (Lean/Coq proofs of individual programs).** Out of scope. LaCaDiLE proves the type system is sound (all programs get Level 1 guarantees). Hull checks the compiler implements the type system. These are system-level proofs. Per-program formal proofs (proving that THIS specific pricing model computes the correct price) are prohibitively expensive and don't scale to AI-generated code. Empirical verification via properties (Level 2) is the practical alternative.

---

## How This Answers the Customer's Concern

**"How do I know the AI-generated code does what I asked?"**

You don't read the generated code. You write properties that define what "correct" means -- price is positive, delta is bounded, the output matches your textbook formula. The toolchain verifies the code against your properties on 100,000 random inputs. If the properties hold, the code is correct (to the confidence level of the random testing). If a property fails, you get the exact input that violates it.

**"How is this different from just writing tests?"**

Tests check specific inputs. Properties check all inputs (empirically, via random sampling). A test says "price(100, 0.2, 0.05, 1.0, 100) = 10.45." A property says "for ALL valid inputs, price is non-negative." The test catches one bug. The property catches every bug in that class. OpenAI showed that 60% of carefully written tests are themselves wrong. Properties are harder to get wrong because they express what, not how.

**"How is this different from a PDF spec that nobody reads?"**

The spec is executable Chelis code. It runs. It produces pass/fail results. It's version-controlled alongside the implementation. It's CI-enforced -- the build fails if a property is violated. It can't drift from the implementation because the toolchain ties them together. A PDF spec sits in a SharePoint folder and rots. An executable property sits in CI and blocks deployment if violated.

**"What if the AI writes wrong properties too?"**

Properties are much simpler than implementations. "Price is positive" is a one-line function. The customer can verify it by reading one line. The implementation of Black-Scholes is 50 lines of optimized tensor code. The failure mode where the AI writes wrong properties AND the properties look correct to the customer is much narrower than the failure mode where the AI writes a wrong implementation.

For the strongest guarantee, the customer writes the properties themselves. They don't need to understand the implementation -- they need to understand their own domain well enough to state invariants. "Put-call parity holds" is something every quant knows. The AI's job is to produce code that satisfies the quant's stated invariants, not to invent the invariants.

---

## Relationship to Existing Plans

**`chelis fuzz` (planned, Phase 3 / pre-Phase 4).** Currently scoped as a CLI tool with simple property flags. This document expands the scope: properties become first-class Chelis functions with `@property` annotations, living in the source tree, CI-enforced, with type-directed random input generation and minimal-counterexample reporting.

**Hull (future shell).** Hull validates the compiler against the language spec via differential testing. The same pattern (reference implementation + production implementation + agreement checking) is what `@property fn matches_reference(...)` does for user code. Hull proves the pattern works on the highest-stakes code in the system (the compiler itself).

**Phase 5g (trusted annotations).** `@convex`, `@lipschitz` start as trusted, evolve toward verified as Beacon (abstract interpretation) matures. No change to the current plan -- this document extends the vision.

**Octant.** LaTeX-to-Deep provenance gives formula traceability. Combined with `@property fn matches_textbook(...)`, the trust chain is: LaTeX formula (human-verified) -> compiled Deep (provenance-linked) -> optimized code (property-verified against the formula). Every link in the chain is machine-checkable.

**CProof.** The commercial pitch incorporates the trust stack directly: "Your quants write pricing models. AI generates optimized code. The compiler guarantees structural soundness. Your quants write domain properties. The toolchain verifies the code satisfies them. You deploy with confidence."

---

## Shells and Tools Affected

| Shell/Tool | Change | Status |
|---|---|---|
| `chelis fuzz` | Evolve from CLI flags to first-class Chelis property functions with `@property`, type-directed input generation, counterexample minimization | Planned, scope expanded by this document |
| Beacon (abstract interpretation) | New future shell: automated static analysis on the tensor DAG, input range specification, overflow/div-zero/NaN detection | Future, not designed |
| `Std.Test` | No change -- `chelis test` remains for deterministic assertion-based tests. `chelis fuzz` is the companion for property-based verification. | Shipped |
| Hull | No change to Hull itself. Hull validates the pattern (differential testing against a reference) that user-facing `@property fn matches_reference` uses. | Future (stub) |
| Phase 5g annotations | No change to near-term plan. Long-term: Beacon may verify annotations automatically. | Deferred |
