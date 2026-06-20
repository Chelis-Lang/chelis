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

The shipped `Effect` enum at `crates/chelis-types/src/types.rs:144-154` has five variants today: `Random`, `Accum`, `Io`, `Test`, and `Resource(String)`. `Random` fires on `dropout`, `uniform_like`, and stdlib random helpers reachable through those operations unless covered by a surrounding `with seed(...)` handler. `Accum` is reserved as an internal design hook for backward-pass accumulation (not yet user-facing). `Io` is narrow today: it covers host-side print and debug only, not network or filesystem access. `Test` is the in-language test runner's effect. `Resource(String)` carries device-resource boundaries (`gpu:0`, `cpu`) and is validated by `chelis build --target {c,hip}` at the build boundary. The set is correct for what it tracks; the next-round expansion of the taxonomy lives in `effect_taxonomy_expansion.md`.

#### Planned expansion

The current taxonomy is bounded but does not yet cover two categories that matter for the broader trust story. Network access and filesystem access are not separate variants today; they fold into either `Io` (in the case of stdout-style writes) or get inferred without effect annotation entirely (in the case of file reads, since no `Std.Io` function declares one yet). The expansion in `effect_taxonomy_expansion.md` is bounded to four items:

- **Item 1** — add `Network` and `Filesystem` variants to the `Effect` enum and annotate every `Std.Io` and shell function that touches them.
- **Item 2** — ship `chelis audit --effects <package.chb>` to surface the union of effects across a package's public API; same data exposed via Tide HTTP and as an MCP `chelis_audit` tool.
- **Item 3** — `chelis run --refuse Network,Filesystem` for signature-based pre-flight refusal of binaries whose declared effects exceed an operator's allowlist (not runtime sandboxing; the binary is still trusted to honestly describe itself).
- **Item 4** — wire effect aggregation into `chelis reef install` so the install path can `--print-effects` and `--refuse` at the package boundary.

Subprocess tracking is intentionally deferred until FFI lands (no current Chelis program can spawn subprocesses; there is nothing to track yet). Signing of artifacts is a separate concern tracked but not in the bounded taxonomy expansion.

These guarantee that a program is internally consistent. They do NOT guarantee it computes the right answer. A program can be dimension-safe, effect-correct, linear, and differentiable while implementing the wrong formula entirely.

### Level 2 -- Executable Properties as Spec (V1 shipped, core value prop)

The user writes executable Chelis functions that define what "correct" means in their domain. These properties ARE the spec. The toolchain verifies the generated code against them empirically on random inputs.

This is the capability that addresses the customer's concern directly: "I can't define a spec and ensure it's actually in the generated code." The answer: "You define the spec as executable properties. We verify the code satisfies them."

**Properties are Chelis functions that return bool:**

```chelis
-- Domain invariants: things that must always be true
@property price_is_positive forall(
    spot: f32, vol: f32, rate: f32, t: f32, strike: f32
) where spot > 0.0, vol > 0.0, t > 0.0, strike > 0.0:
  price(spot, vol, rate, t, strike) > 0.0

@property delta_in_unit_interval forall(
    spot: f32, vol: f32, rate: f32, t: f32, strike: f32
) where spot > 0.0, vol > 0.0, t > 0.0, strike > 0.0:
  (grad(price, wrt=spot)(spot, vol, rate, t, strike) >= 0.0)
    && (grad(price, wrt=spot)(spot, vol, rate, t, strike) <= 1.0)

@property put_call_parity_holds forall(
    spot: f32, vol: f32, rate: f32, t: f32, strike: f32
) where spot > 0.0, vol > 0.0, t > 0.0, strike > 0.0:
  close(
    sub(call_price(spot, vol, rate, t, strike),
        put_price(spot, vol, rate, t, strike)),
    sub(spot, mul(strike, exp(neg(mul(rate, t))))),
    1e-4)

-- Spec correspondence: the complex code matches the simple formula
@property matches_textbook forall(
    spot: f32, vol: f32, rate: f32, t: f32, strike: f32
) where spot > 0.0, vol > 0.0, t > 0.0, strike > 0.0:
  close(
    price(spot, vol, rate, t, strike),
    textbook_black_scholes(spot, vol, rate, t, strike),
    1e-6)
```

The last pattern -- `matches_textbook` -- is the highest-value property type. The user (or their quant) writes a 5-line direct transcription of the formula from a textbook. It's obviously correct by inspection. The AI generates the optimized `price` function (vectorized, fused, GPU-dispatched). The toolchain runs both on deterministic samples from the property binders and verifies they agree.

**The customer doesn't review the generated code. They review the properties.** Properties say what correct means. If the properties are right and the code satisfies them, the code is right. If the AI generates code that violates a property, it doesn't ship.

**Three property categories:**

1. **Domain invariants.** Things that must always be true regardless of implementation: output is non-negative, delta is between 0 and 1, put-call parity holds, portfolio weights sum to 1. These are cheap to state and catch a wide class of implementation bugs.

2. **Spec correspondence.** The optimized implementation matches a simple reference implementation on random inputs. The reference implementation IS the spec -- it's a direct, obviously-correct transcription of the formula. The optimized implementation is what runs in production. Disagreement on any input is a definitive bug.

3. **Behavioral constraints.** Monotonicity (price increases with spot), continuity (small input change produces small output change), symmetry (put-call symmetry), convergence (Monte Carlo converges as path count increases). These encode domain knowledge that test suites can't express.

**Implementation: evolution of `chelis prove`**

Status: V1 shipped in v0.7.1. The `chelis prove` subcommand and the
first-class `@property NAME forall(...) where ...:` Surf syntax are
implemented (`crates/chelis-cli/src/prove.rs`), with deterministic
type-directed sampling, `--samples`/`--seed`/`--only`/`--json` flags, a
four-state exit-code contract (pass 0, fail 1, unsupported 2, error 3),
and Deep-bridge provenance. V1 covers scalar binders and fixed-shape
numeric tensors. Still to land: symbolic-dimension tensor binders and
counterexample minimization. The bullets below describe the full design;
items beyond the V1 scope above are still forthcoming.

`chelis prove` runs first-class property declarations:

- Properties are Chelis functions (not CLI flags). They live in `properties/*.ch` alongside the code, version-controlled, CI-enforced.
- `chelis prove src/pricer.ch` discovers `@property` annotations and tests each on random inputs drawn from the parameter types.
- Random input generation is type-directed. V1 supports scalar binders and fixed-shape numeric tensors.
- Failure on any input produces the failing input as a minimal reproducible case: "property `price_is_positive` violated at spot=0.001, vol=3.5, rate=-0.02, T=0.001, strike=1000.0".
- `chelis prove --samples 1000` controls the number of random inputs. Default: 100.
- CI integration: `chelis prove` runs as a CI gate alongside `chelis test`. Properties must hold on all sampled inputs for the build to pass.

**Differential testing is a special case of property sampling:**

```chelis
@property matches_reference forall(x: tensor[32, f32]):
  close(optimized_fn(x), reference_fn(x), 1e-6)
```

No separate differential testing infrastructure needed. It's a property that compares two implementations. The property runner handles the random input generation and the agreement checking.

**Canonical domain properties ship with domain shells.** Every domain shell includes a `properties/` directory containing reference `@property` functions for the domain's standard invariants. These are not tests (they live alongside `tests/`, not inside it). They are verification contracts that demonstrate the `@property` pattern on real domain code.

For Shoals (finance):

- `properties/pricing.ch` -- put-call parity, price positivity, call bounded by spot, delta in [0,1], gamma positive for vanilla Europeans
- `properties/greeks.ch` -- source-level grad-derived Greeks match textbook Black-Scholes Greeks within tolerance; executable tests retain finite-difference checks until the full pricing body is IR-lowerable under host-runtime `grad`
- `properties/monte_carlo.ch` -- Monte Carlo price converges to analytic price as path count increases, variance decreases with path count
- `properties/no_arbitrage.ch` -- no-arbitrage conditions on option spreads (bull spread payoff non-negative, butterfly spread payoff non-negative)

For Octant (LaTeX bridge):

- `properties/roundtrip.ch` -- `parse(render(expr))` recovers the original expression; `lower(parse(latex))` type-checks
- `properties/provenance.ch` -- every Deep AST node has a provenance span that points inside the original LaTeX string

For future vertical shells: the same pattern. Canonical properties are the first thing a new domain shell ships, alongside the implementation code they verify.

The onboarding story: "Install Shoals. Run `chelis prove src/`. See 15 canonical finance properties pass on the reference implementations. Now write your own pricing model and add your own properties." The properties are documentation-by-example, not a separate product.

The convention is a hard rule: properties are co-located with the implementation they verify, NOT packaged as a separate "properties" shell. The `@property` infrastructure is in the compiler (`chelis prove`); the properties themselves are domain-specific and belong inside the domain shell. A standalone "finance properties" package with no implementation code is an empty vessel.

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

A specialized analysis tool (working name: Hydrostatic) that runs on the compiled tensor DAG and computes over-approximations of value ranges at each node. The user doesn't annotate anything. The tool infers properties automatically.

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

Not currently designed. Recorded as a future shell (Hydrostatic) in the ecosystem. Depends on the tensor DAG being stable (it is) and the input range specification mechanism (needs design).

**Contract evolution:** Phase 5g's trusted annotations (`@convex`, `@lipschitz`) start as documentation. As Hydrostatic matures, it can verify some of these annotations automatically: check whether `@convex` actually holds by analyzing the second derivative's sign over the specified input ranges. The annotations don't change; the verification level increases over time.

### What's NOT in the stack

**Refinement types (SMT-verified predicates on types).** Dropped. They add programming model complexity (users must understand Z3, restructure code to help the solver, debug incomprehensible solver failures) and don't address the actual concern (does the code match intent). Properties expressed as executable Chelis functions and tested empirically are more actionable, more debuggable, and more accessible than SMT-verified type predicates.

**Runtime monitoring as a language feature.** Dropped. If constraints are captured at compile time (Level 1) and empirically verified pre-deployment (Level 2), runtime monitoring inside the program is redundant. Boundary validation (API inputs, database results, external data feeds) is an application concern handled by library code at I/O boundaries, not a language feature.

**Full formal verification (Lean/Coq proofs of individual programs).** Out of scope. LaCaDiLE proves the type system is sound (all programs get Level 1 guarantees). Hull checks the compiler implements the type system. These are system-level proofs. Per-program formal proofs (proving that THIS specific pricing model computes the correct price) are prohibitively expensive and don't scale to AI-generated code. Empirical verification via properties (Level 2) is the practical alternative.

---

## How This Answers the Customer's Concern

**"How do I know the AI-generated code does what I asked?"**

You don't read the generated code. You write properties that define what "correct" means -- price is positive, delta is bounded, the output matches your textbook formula. The toolchain verifies the code against deterministic samples from those properties. If the properties hold, the code is correct to the confidence level of the random testing. If a property fails, you get the exact input that violates it.

**"How is this different from just writing tests?"**

Tests check specific inputs. Properties check all inputs (empirically, via random sampling). A test says "price(100, 0.2, 0.05, 1.0, 100) = 10.45." A property says "for ALL valid inputs, price is non-negative." The test catches one bug. The property catches every bug in that class. OpenAI showed that 60% of carefully written tests are themselves wrong. Properties are harder to get wrong because they express what, not how.

**"How is this different from a PDF spec that nobody reads?"**

The spec is executable Chelis code. It runs. It produces pass/fail results. It's version-controlled alongside the implementation. It's CI-enforced -- the build fails if a property is violated. It can't drift from the implementation because the toolchain ties them together. A PDF spec sits in a SharePoint folder and rots. An executable property sits in CI and blocks deployment if violated.

**"What if the AI writes wrong properties too?"**

Properties are much simpler than implementations. "Price is positive" is a one-line function. The customer can verify it by reading one line. The implementation of Black-Scholes is 50 lines of optimized tensor code. The failure mode where the AI writes wrong properties AND the properties look correct to the customer is much narrower than the failure mode where the AI writes a wrong implementation.

For the strongest guarantee, the customer writes the properties themselves. They don't need to understand the implementation -- they need to understand their own domain well enough to state invariants. "Put-call parity holds" is something every quant knows. The AI's job is to produce code that satisfies the quant's stated invariants, not to invent the invariants.

---

## Relationship to Existing Plans

**`chelis prove` (V1 shipped, v0.7.1).** Originally scoped as a CLI tool with simple property flags; now shipped as the expanded design — properties are first-class Chelis functions with `@property NAME forall(...) where ...:` annotations, living in the source tree, CI-enforced, with deterministic type-directed input generation and a stable exit-code/NDJSON contract. V1 covers scalar binders and fixed-shape numeric tensors; symbolic-dimension tensor binders and minimal-counterexample reporting remain to land.

**Hull (shipped, v0.1.2).** Hull is the compiler-vs-spec differential layer. It is a self-hosted executable specification shell (pure Chelis, depends on `chelis-std` only) that implements the LaCaDiLE typing rules and small-step operational semantics directly as a reference type checker and a reference evaluator over the Deep AST. The differential harness runs both against the real `chelis` compiler on type-directed, well-typed-by-construction generated programs and asserts they agree. What it proves, precisely:

- The Hull reference checker and reference evaluator AGREE with the shipped compiler on 10,000 generated programs, with ZERO `CompilerUnsound` (Hull rejects, compiler accepts, no documented gap) and zero unexplained `Disagree`. Two independent fresh-seed campaigns reproduced this. The eval lane agrees within f32 tolerance on 1,000 eval-eligible programs (NaN reconciled to JSON `null`).
- The generated 10k lane is Hull-accepted-by-construction, so it proves AGREEMENT. The DETECTION capability (that the harness *would* flag a real `CompilerUnsound`) is proven separately by an injected-unsound self-test and by a hand-curated `known_conservative.json` that exercises the reject direction.
- It is CI-enforced in this monorepo: `tests/conformance/hull/` is the frozen, version-stamped corpus and `.github/workflows/conformance.yml` runs the gate per PR and on push-to-main, failing on any `CompilerUnsound`, unexplained `Disagree`, or `EvalDisagree`. The teeth are verified by an injected-unsound step that EXPECTS the gate to fail. The full fresh 10k campaign runs nightly (`conformance-nightly.yml`).

Scope of v0.1.0: the PURE in-fragment surface (tensor/scalar ops, lambda, let, if, match with `PVar`/`PLit`/`PWildcard`, and the surface `grad` check). The same agreement pattern (reference implementation + production implementation + agreement checking) is what `@property matches_reference forall(...)` does for user code; Hull proves it on the highest-stakes code in the system, the compiler itself. Hull covers the compiler/spec layer that `chelis prove` (per-program `@property`) and c-earchin (spec translation) do not. Three documented v0.1.0 boundaries, all scoped for v0.2.0 in Hull's `docs/v0_2_0_roadmap.md`: grad conservatism is a BUILD-differential concern (the compiler's linearity/Δ rejection is at lowering, not at `chelis check`, so it is out of the v0.1.0 check+eval differential); effects and ADTs (`EConstruct`/`PConstruct`) are out of the typed fragment; builtin-name shadowing and division-by-zero are documented reference/UB gaps, not soundness findings.

**Phase 5g (trusted annotations).** `@convex`, `@lipschitz` start as trusted, evolve toward verified as Hydrostatic (abstract interpretation) matures. No change to the current plan -- this document extends the vision.

**Octant.** LaTeX-to-Deep provenance gives formula traceability. Combined with `@property matches_textbook forall(...)`, the trust chain is: LaTeX formula (human-verified) -> compiled Deep (provenance-linked) -> optimized code (property-verified against the formula). Every link in the chain is machine-checkable, and the chain is shipped end-to-end. Octant emits span-attributed Deep + sidecar `.spans.json`; the chelis-side preservation of those spans through IR lowering, transformation passes, and backend codegen lands per `chelis_span_survival.md` (phases S0-S6, all shipped). Both the DAG-routed path (compute-heavy tensor kernels, §2.4) and the host-routed path (pure-scalar / control-flow / scaffolding, §2.4 host-path rules + §2.4.2) preserve spans, and the post-S6 canary in §4 demonstrably runs Black-Scholes scalar form end-to-end: LaTeX byte range -> Deep node -> IR / HostExpr node -> emitted `// span: <id>` comment in the generated C source line.

**CProof.** The commercial pitch incorporates the trust stack directly: "Your quants write pricing models. AI generates optimized code. The compiler guarantees structural soundness. Your quants write domain properties. The toolchain verifies the code satisfies them. You deploy with confidence."

---

## Trust Stack Implications for Editing Tools

The trust stack guarantees apply equally to human-authored and agent-authored
code. A function written by an AI agent and a function written by a quant
both go through the same type checker, the same effect inference, the same
property checks. The compiler doesn't distinguish authorship.

This has implications for editing tools: a structural editing primitive that
modifies Chelis code (changes a function body, renames a symbol, changes
a signature) preserves the trust stack invariants automatically. The edit
either produces type-correct, effect-correct, property-satisfying code or
the compiler rejects it. Agents can iterate on edits with the trust stack
as the validation layer; the trust stack doesn't need separate "is this
edit acceptable" tooling because the compiler already answers that question.

This is one rationale for the exploratory Agent Editing Surface direction
(see `spec/design/chelis_agent_editing_surface.md`). Structural editing
tools are the natural integration point for agent-driven code modification
because they ride the trust stack as their validation surface.

---

## Limits of the current trust stack

This section keeps future readers from claiming more than is built. Each bullet records a property that is sometimes ascribed to the trust stack but is not actually shipped today.

- **No artifact signing beyond SHA256 content addressing.** Reef artifacts are content-addressed (the lockfile pins `archive_sha256` and `shell_sha256`; substituted bytes are detected on install). They are NOT cryptographically signed by any publisher key. Authenticity (who published this) is a separate property from integrity (the bytes are what was pinned), and the trust stack today provides only the latter. Signing is tracked but demand-driven; it is not designed in `effect_taxonomy_expansion.md` or `reef_distribution.md`.
- **No runtime sandboxing.** Effect checking is compile-time only. A binary that has been post-edited (function bodies replaced with arbitrary C, for example) executes whatever it contains; the runtime does not re-verify the manifest against the binary's actual behavior. The Item 3 capability flag (`chelis run --refuse Network,Filesystem`) is signature-based pre-flight refusal: it consults the binary's declared manifest, not its actual syscall pattern. True syscall interception requires OS-level integration and is post-roadmap.
- **No bit-reproducible-build verification end-to-end.** The reef manifest pins source hashes and the compiler version, so the inputs are reproducible. The C emitter has not been audited for non-determinism (timestamps, embedded paths, hostnames). A claim that "two independent rebuilds from the same source produce bit-identical bytes" is plausible from the architecture but not verified end-to-end today.
- **No FFI security model.** FFI does not exist yet. When it lands in a later phase, security becomes a design question for that phase. Until then, the surface that an FFI security model would protect against simply does not exist; Chelis programs cannot reach C, cannot embed-and-execute binary payloads, and cannot route around the type system through string-to-code conversion.
- **Effect taxonomy is narrower than the broader trust pitch suggests.** As detailed above, `Network` and `Filesystem` are not yet variants. The `effect_taxonomy_expansion.md` bounded plan is what closes that gap; until it lands, claims that "Chelis tracks network access at compile time" are roadmap, not shipped.

These limits are stable: each will move from "limit" to "shipped" only when a corresponding item lands and produces a demo-able CLI command. They are NOT the same as future-product aspirations; they are specifically the gap between what is sometimes attributed to the stack and what the stack actually demonstrates today.

## Shells and Tools Affected

| Shell/Tool | Change | Status |
|---|---|---|
| `chelis prove` | Evolve from CLI flags to first-class Chelis property functions with `@property`, type-directed input generation, counterexample minimization | V1 shipped (v0.7.1); symbolic-dim tensor binders + counterexample minimization pending |
| Hydrostatic (abstract interpretation) | New future shell: automated static analysis on the tensor DAG, input range specification, overflow/div-zero/NaN detection | Future, not designed |
| `Std.Test` | No change -- `chelis test` remains for deterministic assertion-based tests. `chelis prove` is the companion for property-based verification. | Shipped |
| Hull | The compiler-vs-spec differential layer. Hull's reference checker + evaluator agree with the shipped compiler on 10k generated programs (zero CompilerUnsound), CI-enforced by `tests/conformance/hull/` + `conformance.yml`. Validates the pattern (differential testing against a reference) that user-facing `@property matches_reference` uses, on the compiler itself. | Shipped (v0.1.2) |
| Phase 5g annotations | No change to near-term plan. Long-term: Hydrostatic may verify annotations automatically. | Deferred |
